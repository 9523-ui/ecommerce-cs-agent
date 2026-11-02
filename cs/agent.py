"""
客服 Agent —— ReAct 循环 + 工具调用轨迹

相比原项目 agent.py 的修正：
1. 用 AsyncOpenAI，真正异步（原版在 async 函数里调同步 client，并发是假的）
2. tool_call_id 使用模型返回的真实 id（原版硬编码 "call_1"，多工具同轮有风险）
3. 同轮多个工具调用合并为一条 assistant 消息 + N 条 tool 消息（符合 OpenAI 协议）
4. 返回结构化 trace，供前端可视化与评估使用
5. 同一会话加锁串行，避免 Web 端连点两次导致 history 交错
6. 按 user 轮次安全裁剪历史，绝不切断 assistant(tool_calls) 与 tool 消息的配对
"""
import asyncio
import json
import re
import time

from openai import AsyncOpenAI

from cs.tools import get_schemas, get_tool_map
from cs.config import CSConfig

MAX_ROUNDS = 5          # 客服典型链路：查 FAQ → 查订单 → 建单，5 轮足够
MAX_TOOL_RESULT = 1800  # 工具结果截断，防止撑爆上下文

SYSTEM_PROMPT = """你是「电商客服 Agent」，一个在线商城的智能客服助手。你的职责是依据平台规则准确回答用户问题，并在能力边界外主动转人工。

【最高原则 · 必须遵守】
1. 一切政策、规则、流程、时效、金额类回答，必须来自 search_faq 的返回内容。禁止凭记忆或常识编造。
2. 如果 search_faq 返回「[未命中]」，说明知识库没有该问题的答案，禁止猜测、禁止给"大致是"、禁止自造政策。此时按问题性质二选一：
   - 问题仍属于商城业务范畴（政策、流程、售后规则）→ 必须调用 handoff_to_human 转人工
   - 问题明显与商城无关（闲聊、写诗、其他领域咨询）→ 礼貌说明服务范围即可，无需转人工
3. 涉及具体订单（物流到哪、退款进度、少件破损）时，必须先向用户索要订单号，再调用 lookup_order。禁止假设或虚构订单状态。
4. 不得承诺知识库中不存在的补偿、赔偿、折扣、加急。注意区分：
   - 质量问题（破损、开胶、少件、错发、功能故障）、补发、换货、维修、催件 → 标准售后服务。
     订单核实后直接建工单；即使用户只说「帮我处理」没讲清具体方案，也先建单交专员跟进，不要转人工。
   - 赔偿、折扣、加急、额外赠品 → 超出标准范围，转人工。
5. 不回答与商城业务无关的问题，不替用户做与购物无关的任务。

【工具结果判断 · 抗误召回】
search_faq 有时会因词面重合返回不相关条目。若返回内容与用户问题明显不是同一件事，
说明知识库实际没有答案：不要拼接无关内容硬答，不要说"根据我们的规定"，
按上面第 2 条处理（业务问题转人工 / 无关问题说明服务范围）。

【工具使用决策】
- 政策/规则咨询 → search_faq
- 订单相关 → 索要订单号 → lookup_order
- 需要后台操作（补发、改派、退款、维修、催件）→ 先建工单 create_ticket（前提：订单号已核实存在）
- 知识库未命中 / 用户明确要求人工 / 涉及资金纠纷与账号安全 / 投诉维权 → handoff_to_human

【多轮上下文】
- 用户在本会话已提供过的信息（订单号、商品、诉求），后续追问中直接沿用，不要重复索要。
- 用户用「它 / 那个 / 这个 / 那…呢」指代上文对象时，结合上文理解，不要反问「您指的是哪个」。
- 用户给出新订单号或切换话题时，以最新信息为准，不要继续沿用旧订单的结论。
- 用户中途改变诉求（如从查物流改为退款），按新诉求重新判断该调哪个工具，不要套用上一轮结论。

【回复风格】
- 简洁口语化，先给结论，再给操作步骤
- 涉及时效和金额，必须引用知识库原文中的数字，不得改写
- 禁止向用户暴露工具名称、相关度分数、知识库编号等内部信息
- 单次回复控制在 150 字以内，条理清晰"""


def _extract_ticket_ids(trace: list[dict]) -> list[str]:
    """从建单工具返回文本中提取真实工单号（而不是错误地返回订单号）"""
    ids = []
    for t in trace:
        if t["tool"] == "create_ticket":
            m = re.search(r"工单号\s*(T\d+)", t["result"])
            if m:
                ids.append(m.group(1))
    return ids


class CsAgent:
    """单会话客服 Agent。多会话隔离由上层（server）维护实例池。"""

    def __init__(self, session_id: str = "default"):
        if not CSConfig.API_KEY:
            raise RuntimeError("未配置 API Key，请检查 .env 中的 DEEPSEEK_API_KEY")
        self.session_id = session_id
        self.client = AsyncOpenAI(api_key=CSConfig.API_KEY, base_url=CSConfig.BASE_URL)
        self.tools = get_tool_map()
        self.schemas = get_schemas()
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._lock = asyncio.Lock()

    # ── 公开接口 ──

    async def chat(self, text: str) -> dict:
        """
        对外入口：同一会话串行执行。
        Web 端用户快速连点两次时，若不加锁，两次请求会交错写入同一份 history，
        导致 assistant(tool_calls) 与 tool 消息错配，API 直接 400。
        """
        async with self._lock:
            return await self._chat(text)

    async def _chat(self, text: str) -> dict:
        """
        返回:
            reply    : 最终回复文本
            trace    : [{"tool","args","result","ms"}, ...] 工具调用轨迹
            rounds   : 消耗的推理轮数
            handoff  : 本次是否转人工
            tickets  : 本次创建的工单号列表
        """
        self.history.append({"role": "user", "content": text})
        trace: list[dict] = []

        for rnd in range(1, MAX_ROUNDS + 1):
            resp = await self.client.chat.completions.create(
                model=CSConfig.MODEL,
                messages=self.history,
                tools=self.schemas,
                temperature=CSConfig.TEMPERATURE,
            )
            msg = resp.choices[0].message

            # 无工具调用 → 得到最终回复
            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                self.history.append({"role": "assistant", "content": reply})
                result = {
                    "reply": reply,
                    "trace": trace,
                    "rounds": rnd,
                    "handoff": any(t["tool"] == "handoff_to_human" for t in trace),
                    "tickets": _extract_ticket_ids(trace),
                }
                self._trim_history()
                return result

            # 有工具调用：一条 assistant（带全部 tool_calls）+ 对应数量的 tool 消息
            self.history.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                t0 = time.time()
                result_text = await self._safe_call(name, args)
                cost = round((time.time() - t0) * 1000)

                trace.append({"tool": name, "args": args, "result": result_text, "ms": cost})
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,          # 用真实 id，不硬编码
                    "content": result_text[:MAX_TOOL_RESULT],
                })

        # 超过最大轮数：兜底转人工，避免死循环烧钱
        reason = "自动处理轮数超限"
        summary = f"用户问题：{text[:100]}，多轮工具调用后仍未得出答案"
        hr = await self._safe_call("handoff_to_human", {"reason": reason, "summary": summary})
        trace.append({
            "tool": "handoff_to_human",
            "args": {"reason": reason, "summary": summary},
            "result": hr,
            "ms": 0,
        })
        over = "抱歉，这个问题我需要请人工同事帮您确认，已为您转接人工客服（工作时间 9:00-22:00）。"
        self.history.append({"role": "assistant", "content": over})
        result = {
            "reply": over,
            "trace": trace,
            "rounds": MAX_ROUNDS,
            "handoff": True,
            "tickets": _extract_ticket_ids(trace),
        }
        self._trim_history()
        return result

    # ── 内部 ──

    def _trim_history(self):
        """
        按 user 轮次裁剪历史。

        关键点：绝不能按固定条数粗暴切片——一旦切在
        assistant(tool_calls) 和其后的 tool 消息之间，下一轮请求会被 API 拒绝。
        user 消息是天然的安全边界，从某个 user 开始保留即可保证每轮完整。
        """
        idxs = [i for i, m in enumerate(self.history) if m.get("role") == "user"]
        if len(idxs) <= CSConfig.HISTORY_MAX:
            return
        cut = idxs[-CSConfig.HISTORY_MAX]
        self.history = [self.history[0]] + self.history[cut:]

    async def _safe_call(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"[工具不存在] {name}，可用工具：{list(self.tools)}"
        try:
            return await tool.execute(**args)
        except TypeError as e:
            return f"[参数错误] 调用 {name} 时参数不匹配：{e}"
        except Exception as e:
            return f"[执行异常] {name}: {type(e).__name__}: {e}"

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
