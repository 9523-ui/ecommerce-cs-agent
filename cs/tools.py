"""
客服工具集 —— 4 个受限工具

设计原则（面试要点）：
1. 不给大模型开放 SQL。客服场景用专用工具，参数化查询，无法越权。
2. 知识库检索带「相关度阈值」，未命中时明确告知模型，使其能决策转人工，
   而不是拿不相关内容硬编答案（幻觉的根源）。
3. 每个工具返回结构化文本，包含失败原因，便于模型自我修正。
"""
import itertools
import json
import os
import sqlite3
from datetime import datetime

from pydantic import BaseModel, Field

from tools.base import BaseTool
from cs.retrieval import search_faq, MATCH_THRESHOLD, get_backend

BASE = os.path.dirname(os.path.abspath(__file__))

# 允许通过环境变量切换数据库：评估脚本用临时副本，避免建单污染演示数据
DB_PATH = os.getenv("CS_DB_PATH") or os.path.join(BASE, "data", "cs.db")

# 进程内自增序列，保证同一毫秒内连续建单也不会主键冲突
_SEQ = itertools.count()

# ── 知识库检索已抽到 cs/retrieval.py ──
# 三种后端（tfidf / embedding / hybrid）通过环境变量 CS_RETRIEVER 切换。
# 各后端阈值由 cs/eval/calibrate.py 用标注数据校准后写入 THRESHOLDS。
# 这里 re-export search_faq / MATCH_THRESHOLD，保持旧调用方完全兼容。


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════
# 工具 1：知识库检索
# ══════════════════════════════════════════
class FaqSearchTool(BaseTool):
    class Params(BaseModel):
        question: str = Field(description="用户原话问题，保留用户的原始用词，不要改写")

    name = "search_faq"
    description = (
        "检索客服知识库，覆盖：退换货政策、物流配送、支付发票、商品咨询、售后保修、会员权益。"
        "凡是涉及平台规则、政策、流程、时效的问题，必须先调用本工具查询，禁止凭记忆回答。"
    )

    async def execute(self, question: str) -> str:
        hits = search_faq(question)
        if not hits or hits[0][1] < MATCH_THRESHOLD:
            best = f"{hits[0][1]:.3f}" if hits else "0.000"
            return (
                f"[未命中] 知识库中没有匹配条目（最高相关度 {best}，命中阈值 {MATCH_THRESHOLD}）。\n"
                "请勿编造政策或承诺。应如实告知用户该问题超出自动客服范围，并调用 handoff_to_human 转人工。"
            )

        blocks = []
        for i, (faq, score) in enumerate(hits, 1):
            if score <= 0:
                continue
            blocks.append(
                f"[{i}] 相关度 {score:.3f} | 分类 {faq['category']} | 编号 {faq['id']}\n"
                f"问：{faq['question']}\n"
                f"答：{faq['answer']}"
            )
        if not blocks:
            return "[未命中] 无有效匹配条目，请转人工。"
        return "\n\n".join(blocks)


# ══════════════════════════════════════════
# 工具 2：订单查询（参数化，不允许自由 SQL）
# ══════════════════════════════════════════
class OrderLookupTool(BaseTool):
    class Params(BaseModel):
        order_no: str = Field(description="订单号，形如 CS20261101001。必须由用户提供，不要猜测")

    name = "lookup_order"
    description = (
        "根据订单号查询：订单状态、商品、金额、是否付款、物流承运商与轨迹、已提交的售后工单。"
        "用户询问「我的订单/发货/到哪了/物流/退款进度」等问题时调用。"
    )

    async def execute(self, order_no: str) -> str:
        order_no = (order_no or "").strip().upper()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_no = ?", (order_no,)
            ).fetchone()
            if not row:
                return (
                    f"[未找到] 订单号 {order_no} 不存在。\n"
                    "请让用户核对订单号（「我的订单」页面可查看，通常以 CS 开头），不要自行猜测或假设订单状态。"
                )

            ship = conn.execute(
                "SELECT * FROM shipments WHERE order_no = ?", (order_no,)
            ).fetchone()
            tickets = conn.execute(
                "SELECT * FROM tickets WHERE order_no = ?", (order_no,)
            ).fetchall()

            lines = [
                "【订单信息】",
                f"订单号：{row['order_no']}",
                f"客户：{row['customer']}（尾号 {row['phone']}）",
                f"商品：{row['product']} × {row['quantity']}",
                f"实付：¥{row['amount']:.2f}",
                f"状态：{row['status']}",
                f"下单时间：{row['created_at']}",
                f"付款状态：{'已付款' if row['paid'] else '未付款'}",
            ]

            if ship:
                lines += [
                    "",
                    "【物流信息】",
                    f"承运商：{ship['carrier']}  运单号：{ship['tracking_no']}",
                    f"最新状态：{ship['state']}（{ship['last_update']}）",
                ]
                if ship["note"]:
                    lines.append(f"异常说明：{ship['note']}")
            else:
                lines += ["", "【物流信息】暂无物流记录（可能尚未发货）"]

            if tickets:
                lines += ["", "【已有售后工单】"]
                for t in tickets:
                    lines.append(
                        f"{t['ticket_id']} | {t['issue_type']} | 状态 {t['status']} | {t['created_at']}"
                    )
            else:
                lines += ["", "【已有售后工单】无"]

            return "\n".join(lines)
        finally:
            conn.close()


# ══════════════════════════════════════════
# 工具 3：创建售后工单
# ══════════════════════════════════════════
ISSUE_TYPES = ["物流异常", "质量问题", "少件错发", "退款问题", "其他"]


class CreateTicketTool(BaseTool):
    class Params(BaseModel):
        order_no: str = Field(description="关联订单号，必须已验证存在")
        issue_type: str = Field(
            description="问题类型，只能是以下之一：物流异常、质量问题、少件错发、退款问题、其他"
        )
        description: str = Field(
            description="问题描述，需包含用户诉求与已核实的关键信息（如物流停滞时长、破损部位）"
        )

    name = "create_ticket"
    description = (
        "创建售后工单。当用户问题需要人工跟进、需要后台操作（补发/退款/改派/维修）时调用。"
        "调用前必须已通过 lookup_order 确认订单号真实存在。"
    )

    async def execute(self, order_no: str, issue_type: str, description: str) -> str:
        order_no = (order_no or "").strip().upper()
        if issue_type not in ISSUE_TYPES:
            return f"[参数错误] issue_type 必须是 {ISSUE_TYPES} 之一，收到的是「{issue_type}」。请修正后重试。"

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            ok = conn.execute(
                "SELECT 1 FROM orders WHERE order_no = ?", (order_no,)
            ).fetchone()
            if not ok:
                return f"[失败] 订单 {order_no} 不存在，无法建单。请先让用户提供正确订单号。"

            # 幂等：同订单同类型且未关闭的工单不重复创建
            dup = conn.execute(
                "SELECT ticket_id FROM tickets WHERE order_no = ? AND issue_type = ? AND status != 'resolved'",
                (order_no, issue_type),
            ).fetchone()
            if dup:
                return (
                    f"[已存在] 该订单已有进行中的「{issue_type}」工单：{dup['ticket_id']}，"
                    "无需重复创建。请告知用户工单已受理及处理时效。"
                )

            # 毫秒 + 进程内序号：纯数字，兼容工单号正则，且并发/高频下不冲突
            ticket_id = "T" + datetime.now().strftime("%Y%m%d%H%M%S%f")[:17] + f"{next(_SEQ) % 100:02d}"
            conn.execute(
                "INSERT INTO tickets (ticket_id, order_no, issue_type, description, status, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (ticket_id, order_no, issue_type, description, "open", _now()),
            )
            conn.commit()
            return (
                f"[建单成功] 工单号 {ticket_id}\n"
                f"订单 {order_no} | 类型 {issue_type} | 状态 open\n"
                "承诺时效：24 小时内专人联系，48 小时内给出处理方案。"
                "请把工单号告知用户，并说明后续会有专员跟进。"
            )
        finally:
            conn.close()


# ══════════════════════════════════════════
# 工具 4：转人工（Agent 的「边界感」）
# ══════════════════════════════════════════
class HandoffTool(BaseTool):
    class Params(BaseModel):
        reason: str = Field(description="转人工原因，如：知识库无匹配、用户明确要求人工、涉及资金纠纷")
        summary: str = Field(
            description="给人工客服的交接摘要：用户诉求 + 已核实的信息（订单号/工单号）+ 已做过的处理，便于人工无缝接手"
        )

    name = "handoff_to_human"
    description = (
        "转接人工客服。以下情况必须调用："
        "(1) 知识库检索未命中，无法依据规则回答；"
        "(2) 用户明确要求人工；"
        "(3) 涉及资金纠纷、账号安全、投诉维权等超出自动客服权限的事项。"
    )

    async def execute(self, reason: str, summary: str) -> str:
        handoff_id = "H" + datetime.now().strftime("%Y%m%d%H%M%S%f")[:17] + f"{next(_SEQ) % 100:02d}"
        # 落盘留痕，便于统计自动解决率（评估指标的数据来源）
        log_path = os.path.join(os.path.dirname(DB_PATH), "handoffs.jsonl")
        record = {
            "handoff_id": handoff_id,
            "time": _now(),
            "reason": reason,
            "summary": summary,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return (
            f"[已转接] 交接单号 {handoff_id}\n"
            "请告知用户：已为其转接人工客服，当前在线客服工作时间为每日 9:00-22:00，"
            f"排队预计 1-3 分钟；并说明交接摘要已同步（{reason}），用户无需重复描述问题。"
        )


# ══════════════════════════════════════════
# 工具注册
# ══════════════════════════════════════════
ALL_TOOLS = [
    FaqSearchTool(),
    OrderLookupTool(),
    CreateTicketTool(),
    HandoffTool(),
]


def get_tool_map() -> dict:
    return {t.name: t for t in ALL_TOOLS}


def get_schemas() -> list:
    return [t.to_openai_schema() for t in ALL_TOOLS]
