"""
改写版端到端验证 —— 检索层的提升是否真的传导到客服回答？

检索指标（Recall@1/3）只是中间量。这里用 10 条口语化改写问题，
分别在 tfidf / embedding / hybrid 后端下跑完整 Agent（真实 LLM 调用），
观察：
  1. 目标 FAQ 是否出现在 search_faq 的返回里（检索是否命中）
  2. 是否触发转人工（未命中兜底被触发 = 用户没得到答案）
  3. 最终回复内容（供人工核查，不靠脆弱的关键词自动打分）

后端切换用 monkeypatch 注入 cs.tools 的模块级 search_faq / MATCH_THRESHOLD，
避免起子进程或 reload 模块；FaqSearchTool.execute 引用的正是这两个全局名。
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cs.tools as T  # noqa: E402
from cs.agent import CsAgent  # noqa: E402
from cs.retrieval import Retriever, THRESHOLDS, FAQ_PATH  # noqa: E402

EVAL = Path(__file__).resolve().parent
REPORT_DIR = EVAL / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FAQS = json.loads(FAQ_PATH.read_text(encoding="utf-8"))["faqs"]
DOCS = [f'{x["question"]} {x["answer"]}' for x in FAQS]
BACKENDS = ("tfidf", "embedding", "hybrid")


def load_queries():
    """每组取一条最口语化的改写（P1），共 10 条"""
    out = []
    for line in (EVAL / "paraphrase.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        out.append((c["id"], c["paraphrases"][0], c["target"]))
    return out


async def run_backend(backend: str, queries):
    r = Retriever(DOCS, FAQS, backend)
    T.search_faq = r.search
    T.MATCH_THRESHOLD = THRESHOLDS[backend]

    rows = []
    for qid, q, target in queries:
        # 每条用独立会话，避免上一题的 history 干扰
        agent = CsAgent(f"e2e-{backend}-{qid}")
        out = await agent.chat(q)
        faq_calls = [t for t in out["trace"] if t["tool"] == "search_faq"]
        result = faq_calls[0]["result"] if faq_calls else ""
        rows.append({
            "qid": qid,
            "question": q,
            "target": target,
            "target_retrieved": target in result,
            "miss": "[未命中]" in result,
            "handoff": out["handoff"],
            "tools": [t["tool"] for t in out["trace"]],
            "reply": out["reply"],
        })
        print(f"  [{backend:9s}] {qid} {'命中' if target in result else '未命中':4s} "
              f"| 转人工 {str(out['handoff']):5s} | {q[:20]}")
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "检索命中率": f"{sum(r['target_retrieved'] for r in rows)/n*100:.1f}%",
        "转人工率": f"{sum(r['handoff'] for r in rows)/n*100:.1f}%",
        "得到政策回答率": f"{sum(r['target_retrieved'] and not r['handoff'] for r in rows)/n*100:.1f}%",
    }


def build_html(all_rows, metrics):
    head = "".join(f"<th>{b}</th>" for b in BACKENDS)
    detail = []
    for i, (qid, q, target) in enumerate(load_queries()):
        cells = []
        for b in BACKENDS:
            r = all_rows[b][i]
            tag = ("<span style='color:#16a34a'>命中</span>" if r["target_retrieved"]
                   else "<span style='color:#dc2626'>未命中</span>")
            cells.append(f"<td>{tag} / {'转人工' if r['handoff'] else '已答'}"
                         f"<div style='color:#64748b;font-size:11px;margin-top:4px'>"
                         f"{r['reply'][:70].replace(chr(10),' ')}</div></td>")
        detail.append(f"<tr><td><b>{qid}</b><div style='font-size:11px;color:#64748b'>{q}</div>"
                      f"<div style='font-size:11px'>目标 {target}</div></td>"
                      f"{''.join(cells)}</tr>")

    mrow = "".join(
        f"<tr><td><b>{k}</b></td>" + "".join(f"<td>{metrics[b][k]}</td>" for b in BACKENDS) + "</tr>"
        for k in ("检索命中率", "转人工率", "得到政策回答率")
    )
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>改写版端到端验证</title><style>
body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#f7f9fc;color:#1e293b;
padding:32px;line-height:1.7}} .wrap{{max-width:1000px;margin:0 auto;background:#fff;
border:1px solid #e2e8f0;border-radius:14px;padding:30px 36px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:26px 0 10px}}
.sub{{color:#64748b;font-size:13px;margin-bottom:20px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#f1f5f9}} .note{{background:#fffbeb;border-left:4px solid #f59e0b;
padding:12px 16px;font-size:13px;margin:16px 0;border-radius:6px}}
</style></head><body><div class="wrap">
<h1>改写版端到端验证</h1>
<div class="sub">生成时间 {datetime.now():%Y-%m-%d %H:%M:%S} · 10 条口语化改写问题 ×
3 种检索后端 · 真实 LLM 调用 · 每条独立会话</div>
<h2>汇总</h2>
<table><thead><tr><th>指标</th>{head}</tr></thead><tbody>{mrow}</tbody></table>
<div class="note">检索指标（Recall）只是中间量；这里验证「换个说法问」时用户是否真的
拿到了政策答案。检索未命中会触发转人工，因此「得到政策回答率」是最终业务指标。<br>
<b>PP09 说明</b>：「收到的东西少了一件」三个后端都没有调用检索工具，而是按 prompt 规则
主动索要订单号（具体订单售后应先核实），属于正确流程，不是检索失败 —— 检索层单独测时
Embedding 的 top1 就是目标 F024。按「不改用例迁就指标」的原则，此处保留原始 10 条数字，
不事后剔除该样本。</div>
<h2>逐条明细</h2>
<table><thead><tr><th>问题（目标 FAQ）</th>{head}</tr></thead><tbody>{''.join(detail)}</tbody></table>
</div></body></html>"""


async def main():
    queries = load_queries()
    print(f"改写问题 {len(queries)} 条 | 后端 {list(BACKENDS)}\n")
    all_rows, metrics = {}, {}
    for b in BACKENDS:
        print(f"=== {b} ===")
        all_rows[b] = await run_backend(b, queries)
        metrics[b] = summarize(all_rows[b])
        print(f"  -> {metrics[b]}\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"meta": {"time": stamp, "queries": len(queries), "thresholds": THRESHOLDS},
               "metrics": metrics, "details": all_rows}
    jp = REPORT_DIR / f"paraphrase_e2e_{stamp}.json"
    hp = REPORT_DIR / f"paraphrase_e2e_{stamp}.html"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hp.write_text(build_html(all_rows, metrics), encoding="utf-8")
    print(f"JSON: {jp}\nHTML: {hp}")


if __name__ == "__main__":
    asyncio.run(main())
