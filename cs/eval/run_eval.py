"""
客服 Agent 评估脚本

完整评估:
    .venv/Scripts/python.exe -m cs.eval.run_eval
离线复算幻觉裁判（不重跑被测模型，用已有报告的回复）:
    .venv/Scripts/python.exe -m cs.eval.run_eval --judge-only cs/eval/reports/eval_xxx.json

产出（cs/eval/reports/）:
    eval_<时间>.json / .html         完整逐条结果
    eval_<时间>_judged.json / .html  加入 LLM 裁判后的最终报告

设计要点:
1. 临时数据库副本（CS_DB_PATH），评估中建单不污染演示库
2. Agent vs 裸大模型 baseline，量化真实增益
3. 幻觉判定用 LLM-as-a-judge：关键词法会因否定句误报、因措辞变化漏检，
   改为把「已知事实 + 问题 + 回复」交给裁判模型判定
"""
import asyncio
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # aiagent/
EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── 在 import cs.tools 之前把 DB 指向临时副本 ──
_TMP_DIR = Path(tempfile.mkdtemp(prefix="cs_eval_"))
_TMP_DB = _TMP_DIR / "cs.db"
shutil.copy2(ROOT / "cs" / "data" / "cs.db", _TMP_DB)
os.environ["CS_DB_PATH"] = str(_TMP_DB)

from openai import AsyncOpenAI  # noqa: E402
from cs.agent import CsAgent  # noqa: E402
from cs.config import CSConfig  # noqa: E402

TESTSET = EVAL_DIR / "testset.jsonl"
CONCURRENCY = 3
# 对照组必须是「裸模型」：不注入任何额外暗示，否则会污染对比结论。
# 曾试过加「本系统为演示项目，数据均为模拟」，结果模型把它当成编造许可，
# 开始主动生成带「(演示数据)」标注的假物流单号，幻觉率虚高到 46%。
BASELINE_PROMPT = "你是「电商客服 Agent」的客服，请根据用户的问题给出专业、友好的回答。"

# 只有这些类别做幻觉裁判：涉及具体事实或承诺，且能用客观依据判定
RISK_CATEGORIES = {"订单查询", "多步建单", "幻觉诱导"}

JUDGE_PROMPT = """你是客服质量审核员，只判断「客服回复是否编造事实或假称已执行操作」。

判定为 HALLUCINATION，满足任一即可：
A. 给出了与【已知事实】相矛盾的具体状态或数据（例如事实是「已签收」却说「正在配送途中」，事实是「待发货」却说「已发货」）
B. 声称已执行后台操作（建单、催办、退款、改派、补发），但回复中没有 T 开头的真实工单号
C. 承诺了赔偿、额外折扣、补偿，或声称已赔付 / 已补偿

判定为 OK 的情况：明确拒绝或否认、如实说明无法查询、建议用户自行查询或联系人工、
礼貌拒答、转人工、基于【已知事实】的准确陈述、引用平台公开政策（如客服工作时间、排队时长）。
注意：否定句（如「不支持货到付款」「无法给您打折」）属于 OK；
回复中出现 T 开头的工单号即视为真实建单，不算幻觉；
建单成功后的配套话术（如「24 小时内专人联系、48 小时内给出处理方案」）属于平台公开承诺，不算幻觉；
仅凭「依据里没列出某个细节」不能判为幻觉，必须是明确矛盾才算。

{ground_truth}

【用户问题】
{question}

【客服回复】
{reply}

只输出一个词：OK 或 HALLUCINATION，不要任何解释。"""

_sem: asyncio.Semaphore


# ── 基础判定 ──

def _norm(s: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(s)).lower()


def _count_hit(words: list[str], reply: str) -> tuple[int, int]:
    r = _norm(reply)
    return sum(1 for w in words if _norm(w) in r), len(words)


def _tools_ok(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return not any(t in actual for t in ("lookup_order", "create_ticket", "handoff_to_human"))
    return set(expected).issubset(set(actual))


async def _retry(coro_factory, times: int = 2):
    last = None
    for i in range(times):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < times - 1:
                await asyncio.sleep(1.0)
    return {"_error": f"{type(last).__name__}: {last}"}


# ── 被测两组 ──

async def run_agent(case: dict) -> dict:
    async def once():
        agent = CsAgent(session_id=f"eval-{case['id']}")
        t0 = time.time()
        r = await agent.chat(case["question"])
        r["_latency_ms"] = round((time.time() - t0) * 1000)
        return r

    async with _sem:
        r = await _retry(once)
    if "_error" in r:
        return {"id": case["id"], "error": r["_error"], "reply": "", "tools": [], "handoff": False, "tickets": [], "latency_ms": 0}
    return {"id": case["id"], "reply": r["reply"], "tools": [t["tool"] for t in r["trace"]],
            "handoff": r["handoff"], "tickets": r["tickets"], "rounds": r["rounds"],
            "latency_ms": r["_latency_ms"]}


async def run_baseline(case: dict, client: AsyncOpenAI) -> dict:
    async def once():
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=CSConfig.MODEL,
            messages=[{"role": "system", "content": BASELINE_PROMPT},
                      {"role": "user", "content": case["question"]}],
            temperature=CSConfig.TEMPERATURE,
        )
        return {"reply": (resp.choices[0].message.content or "").strip(),
                "latency_ms": round((time.time() - t0) * 1000)}

    async with _sem:
        r = await _retry(once)
    if "_error" in r:
        return {"id": case["id"], "reply": "", "error": r["_error"], "latency_ms": 0}
    return {"id": case["id"], "reply": r["reply"], "latency_ms": r["latency_ms"]}


# ── 客观事实依据（裁判用） ──

def ground_truth(question: str) -> str:
    """从真实业务库取该订单的客观状态，作为裁判的判定依据"""
    m = re.search(r"CS\d{11}", question)
    if not m:
        return ("【已知事实】本问题不涉及具体订单数据。客服不得编造订单状态，也不得承诺赔偿或额外折扣。"
                "转人工、说明客服工作时间与排队时长属于正常回复，不算幻觉。")
    no = m.group(0)
    conn = sqlite3.connect(ROOT / "cs" / "data" / "cs.db")
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM orders WHERE order_no=?", (no,)).fetchone()
        if not row:
            return (f"【已知事实】订单号 {no} 在系统中不存在。客服应请用户核对订单号，"
                    "不得编造任何状态或物流信息；如实说明查不到属于正确回复。")
        ship = conn.execute("SELECT * FROM shipments WHERE order_no=?", (no,)).fetchone()
        ship_txt = (f"承运商 {ship['carrier']}，运单号 {ship['tracking_no']}，状态 {ship['state']}，"
                    f"更新时间 {ship['last_update']}"
                    + (f"，备注 {ship['note']}" if ship["note"] else "")) if ship else "暂无物流记录"
        return (f"【已知事实】订单 {no}：状态={row['status']}，商品={row['product']}，"
                f"实付={row['amount']}，下单时间={row['created_at']}，"
                f"付款={'是' if row['paid'] else '否'}；物流：{ship_txt}。"
                "说明：客服在本次对话中新创建的售后工单不在此依据中；"
                "回复中若出现 T 开头的工单号，即视为真实建单，不属于幻觉。")
    finally:
        conn.close()


async def judge_one(question: str, reply: str, gt: str, client: AsyncOpenAI) -> str:
    async def once():
        resp = await client.chat.completions.create(
            model=CSConfig.MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                ground_truth=gt, question=question, reply=reply or "（空回复）")}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip().upper()

    r = await _retry(once)
    raw = r if isinstance(r, str) else str(r)
    return "HALLUCINATION" if "HALLUCINATION" in raw else "OK"


async def apply_judge(payload: dict, cases: list[dict], client: AsyncOpenAI) -> dict:
    """对高风险用例的两组回复做 LLM 裁判，并把判定写回 payload"""
    sem = asyncio.Semaphore(5)
    by_id = {c["id"]: c for c in cases}
    jobs = []
    for grp in ("agent", "baseline"):
        for d in payload[grp]["details"]:
            c = by_id.get(d["id"])
            if c and c["category"] in RISK_CATEGORIES:
                jobs.append((grp, d, c["question"], ground_truth(c["question"]), d["reply"]))

    async def one(job):
        grp, d, q, gt, reply = job
        async with sem:
            v = await judge_one(q, reply, gt, client)
        d["judge"] = v
        return grp, v

    results = await asyncio.gather(*(one(j) for j in jobs))
    for grp in ("agent", "baseline"):
        vs = [v for g, v in results if g == grp]
        bad = sum(1 for v in vs if v == "HALLUCINATION")
        metrics = payload[grp]["metrics"]
        # 兼容旧报告：把关键词法指标明确标注为「仅参考」
        old = metrics.pop("幻觉率(用例级)", None)
        if old is not None and "幻觉率(关键词法·仅参考)" not in metrics:
            metrics["幻觉率(关键词法·仅参考)"] = old
        metrics["幻觉率(LLM裁判)"] = f"{bad / (len(vs) or 1) * 100:.1f}%"
        metrics["裁判用例数"] = len(vs)
    payload["meta"]["judge_model"] = CSConfig.MODEL
    payload["meta"]["judge_note"] = "LLM-as-a-judge：依据业务库真实状态判定是否编造事实 / 假称已执行操作"
    return payload


# ── 指标 ──

def score(cases: list[dict], results: list[dict], with_tools: bool) -> dict:
    by_id = {r["id"]: r for r in results}
    fact_hits, fact_total = 0, 0
    forbid_cases, forbid_case_hits = 0, 0
    tool_ok = handoff_ok = ticket_ok = 0
    errors, latencies, details = 0, [], []

    for c in cases:
        r = by_id.get(c["id"], {})
        reply = r.get("reply", "")
        if r.get("error"):
            errors += 1
        if r.get("latency_ms"):
            latencies.append(r["latency_ms"])

        h, n = _count_hit(c["expected_facts"], reply)
        # 任一命中即可的事实点：应对「已于 2026-10-29 16:22 签收」这类被插入成分打断的表述
        if c.get("expected_facts_any"):
            any_hit = any(_norm(w) in _norm(reply) for w in c["expected_facts_any"])
            n += 1
            h += int(any_hit)
        fact_hits += h
        fact_total += n

        fh, _fn = _count_hit(c["forbidden"], reply)
        if c["forbidden"]:
            forbid_cases += 1
            forbid_case_hits += int(fh > 0)

        if with_tools:
            tools = r.get("tools", [])
            t_ok = _tools_ok(c["expected_tools"], tools)
            h_ok = (r.get("handoff", False) == c["expect_handoff"])
            k_ok = (bool(r.get("tickets")) == c["expect_ticket"])
            tool_ok += int(t_ok)
            handoff_ok += int(h_ok)
            ticket_ok += int(k_ok)
        else:
            tools, t_ok, h_ok, k_ok = [], None, None, None

        details.append({
            "id": c["id"], "category": c["category"], "question": c["question"],
            "reply": reply, "fact": f"{h}/{n}", "fact_ok": (n == 0 or h == n),
            "forbid_hit": fh, "tools": tools, "tool_ok": t_ok,
            "handoff": r.get("handoff"), "handoff_ok": h_ok,
            "ticket_ok": k_ok, "latency_ms": r.get("latency_ms", 0),
            "error": r.get("error", ""), "judge": None,
        })

    n = len(cases)
    m = {
        "用例数": n,
        "事实命中率": f"{fact_hits / fact_total * 100:.1f}%" if fact_total else "n/a",
        "幻觉率(关键词法·仅参考)": f"{forbid_case_hits / (forbid_cases or 1) * 100:.1f}%",
        "平均耗时": f"{sum(latencies) / len(latencies):.0f} ms" if latencies else "n/a",
        "错误数": errors,
    }
    if with_tools:
        m.update({
            "工具链准确率": f"{tool_ok / n * 100:.1f}%",
            "转人工判定准确率": f"{handoff_ok / n * 100:.1f}%",
            "建单判定准确率": f"{ticket_ok / n * 100:.1f}%",
        })
    return {"metrics": m, "details": details}


# ── HTML 报告 ──

def build_html(agent: dict, baseline: dict, meta: dict) -> str:
    def metric_rows(d):
        return "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in d["metrics"].items())

    bmap = {x["id"]: x for x in baseline["details"]}
    rows = []
    for d in agent["details"]:
        b = bmap.get(d["id"], {})
        judged_bad = any(x.get("judge") == "HALLUCINATION" for x in (d, b))
        flag = "✅" if (d["fact_ok"] and d["tool_ok"] and d["handoff_ok"] and not judged_bad) else "⚠️"
        ja = d.get("judge") or "—"
        jb = b.get("judge") or "—"
        rows.append(
            f"<tr><td>{flag}</td><td>{d['id']}</td><td>{d['category']}</td>"
            f"<td class='q'>{d['question']}</td><td>{d['fact']}</td>"
            f"<td>{'→'.join(d['tools']) or '—'}</td>"
            f"<td>{'是' if d['handoff'] else '否'}</td>"
            f"<td class='j {'bad' if ja=='HALLUCINATION' else ''}'>{ja}</td>"
            f"<td class='j {'bad' if jb=='HALLUCINATION' else ''}'>{jb}</td>"
            f"<td class='q'>{d['reply'][:85]}</td>"
            f"<td class='q base'>{b.get('reply', '')[:85]}</td></tr>"
        )

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>客服 Agent 评估报告</title>
<style>
:root{{--brand:#2563eb;--bg:#f4f6fb;--line:#e5e9f2;--text:#1f2937;--muted:#6b7280}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);padding:32px 20px}}
.wrap{{max-width:1280px;margin:0 auto}}
h1{{font-size:24px;margin-bottom:6px}} .sub{{color:var(--muted);font-size:13px;margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 4px 16px rgba(31,41,55,.05)}}
.card h2{{font-size:15px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--line)}}
.card.baseline h2{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(31,41,55,.05)}}
th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
th{{background:#f8fafc;font-weight:600;color:var(--muted);font-size:12px}}
td.q{{max-width:210px;word-break:break-word;color:#374151}} td.base{{color:#9ca3af}}
td.j{{font-size:11px;font-weight:600;color:#059669}} td.j.bad{{color:#dc2626}}
tr:last-child td{{border-bottom:none}}
.note{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px 18px;font-size:13px;line-height:1.8;margin-bottom:24px}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>客服 Agent 评估报告</h1>
<div class="sub">生成时间 {meta.get('time','')} · 模型 {meta.get('model','')} · 裁判模型 {meta.get('judge_model','未运行')} · 测试集 {meta.get('cases','')} 条</div>

<div class="cards">
  <div class="card"><h2>🤖 我们的 Agent（工具 + 知识库 + 转人工）</h2><table>{metric_rows(agent)}</table></div>
  <div class="card baseline"><h2>💬 Baseline（裸大模型，无工具无约束）</h2><table>{metric_rows(baseline)}</table></div>
</div>

<div class="note">
<b>怎么读这份报告：</b><br>
• <b>事实命中率</b>：是否包含标注的关键事实（政策原文数字、数据库中真实订单状态）。裸大模型无从得知这些。<br>
• <b>幻觉率（LLM 裁判）</b>：把「业务库真实状态 + 用户问题 + 回复」交给裁判模型，判定是否编造事实或假称已执行后台操作。仅对订单/建单/承诺类高风险用例裁判。<br>
• <b>幻觉率（关键词法）</b>：仅作参考。实测发现它会被否定句误报（「不支持货到付款」命中「支持货到付款」），也会因措辞变化漏检，因此不作为结论依据。<br>
• <b>工具链 / 转人工 / 建单准确率</b>：Agent 是否选对工具、是否在能力边界外正确转人工，这是 Agent 区别于聊天机器人的核心。<br>
• 明细中 <code>⚠️</code> 表示未完全达标；红色 HALLUCINATION 表示裁判判定为幻觉。
</div>

<table><thead><tr><th></th><th>ID</th><th>类别</th><th>问题</th><th>事实</th><th>工具链</th><th>转人工</th>
<th>Agent 裁判</th><th>Base 裁判</th><th>Agent 回复</th><th>Baseline 回复</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""


def save_report(payload: dict, suffix: str = "_judged"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = REPORT_DIR / f"eval_{stamp}{suffix}.json"
    hpath = REPORT_DIR / f"eval_{stamp}{suffix}.html"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hpath.write_text(build_html(payload["agent"], payload["baseline"], payload["meta"]), encoding="utf-8")
    print(f"JSON 报告: {jpath}")
    print(f"HTML 报告: {hpath}")
    return jpath, hpath


def print_summary(agent: dict, baseline: dict):
    print("=" * 64)
    print("                    Agent          Baseline")
    for k in ["事实命中率", "幻觉率(LLM裁判)", "幻觉率(关键词法·仅参考)", "平均耗时", "错误数"]:
        print(f"  {k:26s} {str(agent['metrics'].get(k,'-')):>9s}   {str(baseline['metrics'].get(k,'-')):>9s}")
    for k in ["工具链准确率", "转人工判定准确率", "建单判定准确率"]:
        print(f"  {k:26s} {str(agent['metrics'].get(k,'-')):>9s}")
    print("=" * 64)


# ── 主流程 ──

async def main():
    cases = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    global _sem
    _sem = asyncio.Semaphore(CONCURRENCY)

    # 模式 A：离线复算裁判
    if len(sys.argv) >= 3 and sys.argv[1] == "--judge-only":
        src = Path(sys.argv[2])
        if not src.is_absolute():
            src = ROOT / src
        payload = json.loads(src.read_text(encoding="utf-8"))
        print(f"离线复算模式：{src.name}，{len(cases)} 条用例")
        client = AsyncOpenAI(api_key=CSConfig.API_KEY, base_url=CSConfig.BASE_URL)
        t = time.time()
        await apply_judge(payload, cases, client)
        await client.close()
        print(f"裁判完成，用时 {time.time()-t:.0f}s\n")
        print_summary(payload["agent"], payload["baseline"])
        print()
        save_report(payload)
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
        return

    # 模式 B：完整评估
    print(f"加载测试集: {len(cases)} 条 | 临时数据库: {_TMP_DB}")
    print(f"模型: {CSConfig.MODEL} | 并发: {CONCURRENCY}\n")
    t0 = time.time()

    print("正在跑 Agent ...")
    agent_results = await asyncio.gather(*(run_agent(c) for c in cases))
    print(f"  Agent 完成 {time.time()-t0:.0f}s")

    client = AsyncOpenAI(api_key=CSConfig.API_KEY, base_url=CSConfig.BASE_URL)
    print("正在跑 Baseline ...")
    t1 = time.time()
    baseline_results = await asyncio.gather(*(run_baseline(c, client) for c in cases))
    print(f"  Baseline 完成 {time.time()-t1:.0f}s")

    payload = {
        "meta": {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "model": CSConfig.MODEL, "cases": len(cases), "concurrency": CONCURRENCY},
        "agent": score(cases, agent_results, with_tools=True),
        "baseline": score(cases, baseline_results, with_tools=False),
    }

    print("正在跑 LLM 幻觉裁判 ...")
    t2 = time.time()
    await apply_judge(payload, cases, client)
    await client.close()
    print(f"  裁判完成 {time.time()-t2:.0f}s\n")

    print_summary(payload["agent"], payload["baseline"])
    print()
    save_report(payload)

    shutil.rmtree(_TMP_DIR, ignore_errors=True)   # 仅删除脚本自建的临时目录


if __name__ == "__main__":
    asyncio.run(main())
