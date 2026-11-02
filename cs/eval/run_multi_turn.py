"""
多轮对话评估 —— 检验 Agent 的上下文继承能力

与 run_eval.py 的区别:
- run_eval.py 每条用例是独立单轮，Agent 每次新建实例
- 本脚本每条用例是「一段连续对话」，同一个 CsAgent 实例贯穿所有轮次，
  因此能测到指代消解、信息补全、任务切换、实体切换这些单轮测不出来的能力

指标定义:
1. 事实命中率      : 各轮标注事实是否出现在回复中
2. 上下文继承率     : 明确要求「不得重复索要已提供信息」的轮次通过率（核心指标）
3. 工具选择准确率   : 各轮是否调用了期望工具 / 是否避免了不该调用的工具
4. 全轮通过率      : 整条用例所有轮次全部达标的比例
5. 禁止词命中      : 编造事实、越界作答的次数

运行:
    .venv/Scripts/python.exe -m cs.eval.run_multi_turn
"""
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 临时库副本：多轮建单不污染演示库
_TMP_DIR = Path(tempfile.mkdtemp(prefix="cs_mt_"))
_TMP_DB = _TMP_DIR / "cs.db"
shutil.copy2(ROOT / "cs" / "data" / "cs.db", _TMP_DB)
os.environ["CS_DB_PATH"] = str(_TMP_DB)

from openai import AsyncOpenAI  # noqa: E402

from cs.agent import CsAgent  # noqa: E402
from cs.config import CSConfig  # noqa: E402

MULTI_SET = EVAL_DIR / "multi_turn.jsonl"

# 对照组同样是裸模型，仅保留中性身份，不注入任何额外暗示
BASELINE_PROMPT = "你是「电商客服 Agent」的客服，请根据用户的问题给出专业、友好的回答。"

# 重复索要已提供信息的话术特征
REASK_PATTERNS = [
    "请提供订单号", "麻烦提供订单号", "请告诉我订单号", "提供一下订单号",
    "请提供您的订单号", "订单号是多少", "方便提供订单号", "需要您的订单号",
]


def _norm(s: str) -> str:
    import re
    return re.sub(r"[\s\u3000]+", "", str(s)).lower()


def _hit_any(words: list[str], reply: str) -> tuple[bool, str]:
    r = _norm(reply)
    for w in words or []:
        if _norm(w) in r:
            return True, w
    return False, ""


def _hit_all(words: list[str], reply: str) -> tuple[int, int, list[str]]:
    r = _norm(reply)
    miss = [w for w in (words or []) if _norm(w) not in r]
    return len(words or []) - len(miss), len(words or []), miss


def score_turn(t: dict, r: dict) -> dict:
    """对单轮打分。返回结构化判定，供汇总与报告使用。"""
    reply = r.get("reply", "")
    tools = r.get("tools", [])

    # 1) 必须全部出现的事实
    h_all, n_all, miss_all = _hit_all(t.get("expected_facts"), reply)
    # 2) 任一出现即可的事实（应对同义表述，如「16:22」和「下午4点22」）
    any_ok, any_word = _hit_any(t.get("expected_facts_any"), reply)
    n_any = 1 if t.get("expected_facts_any") else 0

    fact_ok = (h_all == n_all) and (any_ok or n_any == 0)

    # 3) 禁止词
    forbid_hit = [w for w in (t.get("forbidden") or []) if _norm(w) in _norm(reply)]

    # 4) 工具约束
    tools_ok = None
    if t.get("expect_tools_any"):
        tools_ok = bool(set(t["expect_tools_any"]) & set(tools))
    if t.get("forbid_tools"):
        bad = set(t["forbid_tools"]) & set(tools)
        tools_ok = (tools_ok is not False) and not bad

    # 5) 是否重复索要已提供的信息
    no_reask_ok = None
    reask_hit = []
    if t.get("expect_no_reask"):
        reask_hit = [p for p in REASK_PATTERNS if _norm(p) in _norm(reply)]
        no_reask_ok = not reask_hit

    # 6) 幂等场景：上一轮已建单，本轮不重复建但复用了真实工单号 —— 等同于完成任务
    #    （建单幂等是刻意设计：用户反复追问不应刷出多张工单）
    if t.get("accept_ticket_id") and re.search(r"T\d{10,}", reply):
        if tools_ok is False:
            tools_ok = True

    ok = fact_ok and not forbid_hit and tools_ok is not False and no_reask_ok is not False

    return {
        "ok": ok, "fact_ok": fact_ok,
        "fact_detail": f"{h_all}/{n_all}" if n_all else ("any" if n_any else "—"),
        "fact_miss": miss_all, "fact_any": any_word,
        "forbid_hit": forbid_hit, "tools_ok": tools_ok,
        "no_reask_ok": no_reask_ok, "reask_hit": reask_hit,
    }


async def run_agent_case(case: dict) -> dict:
    """同一实例贯穿多轮 —— 这是多轮评估成立的前提"""
    agent = CsAgent(session_id=f"mt-{case['id']}")
    turns = []
    for t in case["turns"]:
        t0 = time.time()
        r = await agent.chat(t["question"])
        turns.append({
            "question": t["question"],
            "reply": r["reply"],
            "tools": [x["tool"] for x in r["trace"]],
            "handoff": r["handoff"],
            "tickets": r["tickets"],
            "ms": round((time.time() - t0) * 1000),
        })
    return {"id": case["id"], "turns": turns}


async def run_baseline_case(case: dict, client: AsyncOpenAI) -> dict:
    """裸模型也给足上下文（自己维护 messages），否则对比不公平"""
    msgs = [{"role": "system", "content": BASELINE_PROMPT}]
    turns = []
    for t in case["turns"]:
        msgs.append({"role": "user", "content": t["question"]})
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=CSConfig.MODEL, messages=msgs, temperature=CSConfig.TEMPERATURE,
        )
        reply = (resp.choices[0].message.content or "").strip()
        msgs.append({"role": "assistant", "content": reply})
        turns.append({
            "question": t["question"], "reply": reply, "tools": [],
            "handoff": False, "tickets": [],
            "ms": round((time.time() - t0) * 1000),
        })
    return {"id": case["id"], "turns": turns}


def summarize(cases: list[dict], runs: dict, with_tools: bool) -> dict:
    by_id = {r["id"]: r for r in runs}
    fact_hit = fact_total = 0
    reask_pass = reask_total = 0
    tool_pass = tool_total = 0
    forbid_cnt = 0
    case_all_ok = 0
    latencies = []

    for c in cases:
        run = by_id.get(c["id"], {"turns": []})
        case_ok = True
        for i, t in enumerate(c["turns"]):
            r = run["turns"][i] if i < len(run["turns"]) else {"reply": "", "tools": [], "ms": 0}
            s = score_turn(t, r)
            if r.get("ms"):
                latencies.append(r["ms"])

            n_all = len(t.get("expected_facts") or [])
            fact_hit += (n_all - len(s["fact_miss"]))
            fact_total += n_all
            if t.get("expected_facts_any"):
                fact_total += 1
                fact_hit += int(bool(s["fact_any"]))

            if t.get("expect_no_reask"):
                reask_total += 1
                reask_pass += int(bool(s["no_reask_ok"]))

            if with_tools and (t.get("expect_tools_any") or t.get("forbid_tools")):
                tool_total += 1
                tool_pass += int(s["tools_ok"] is True)

            forbid_cnt += len(s["forbid_hit"])
            if not s["ok"]:
                case_ok = False
        case_all_ok += int(case_ok)

    m = {
        "用例数": len(cases),
        "总轮次": sum(len(c["turns"]) for c in cases),
        "事实命中率": f"{fact_hit / (fact_total or 1) * 100:.1f}%" if fact_total else "n/a",
        "禁止词命中次数": forbid_cnt,
        "全轮通过率": f"{case_all_ok / (len(cases) or 1) * 100:.1f}%",
        "平均单轮耗时": f"{sum(latencies) / (len(latencies) or 1):.0f} ms",
    }
    if with_tools:
        m["上下文继承率"] = f"{reask_pass / (reask_total or 1) * 100:.1f}%" if reask_total else "n/a"
        m["工具选择准确率"] = f"{tool_pass / (tool_total or 1) * 100:.1f}%" if tool_total else "n/a"
    return {"metrics": m, "details": details_of(cases, runs, with_tools)}


def details_of(cases: list[dict], runs: dict, with_tools: bool) -> list[dict]:
    by_id = {r["id"]: r for r in runs}
    out = []
    for c in cases:
        run = by_id.get(c["id"], {"turns": []})
        rows = []
        for i, t in enumerate(c["turns"]):
            r = run["turns"][i] if i < len(run["turns"]) else {"reply": "", "tools": [], "ms": 0}
            s = score_turn(t, r)
            rows.append({"turn": i + 1, "question": t["question"], "reply": r["reply"],
                         "tools": r["tools"], "handoff": r.get("handoff", False),
                         "score": s})
        out.append({"id": c["id"], "name": c["name"], "capability": c["capability"],
                    "turns": rows})
    return out


def build_html(agent: dict, baseline: dict, meta: dict) -> str:
    def metric_rows(d):
        return "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in d["metrics"].items())

    blocks = []
    bmap = {x["id"]: x for x in baseline["details"]}
    for c in agent["details"]:
        b = bmap.get(c["id"])
        rows = []
        for i, t in enumerate(c["turns"]):
            s = t["score"]
            bt = b["turns"][i] if b and i < len(b["turns"]) else None
            bs = bt["score"] if bt else None
            flag = "✅" if s["ok"] else "⚠️"
            mark = " · ".join(filter(None, [
                "" if s["fact_ok"] else "事实缺失",
                "禁止词:" + ",".join(s["forbid_hit"]) if s["forbid_hit"] else "",
                "工具不符" if s["tools_ok"] is False else "",
                "重复索要:" + ",".join(s["reask_hit"]) if s["reask_hit"] else "",
            ]))
            rows.append(
                f"<tr><td>{flag}</td><td>{t['turn']}</td><td class='q'>{t['question']}</td>"
                f"<td>{'→'.join(t['tools']) or '—'}</td>"
                f"<td class='q'>{t['reply'][:150]}</td>"
                f"<td class='m'>{mark or '通过'}</td>"
                f"<td class='q base'>{bt['reply'][:110] if bt else ''}</td></tr>"
            )
        blocks.append(
            f"<h2>{c['id']} · {c['name']} <span class='cap'>{c['capability']}</span></h2>"
            f"<table><thead><tr><th></th><th>轮</th><th>用户说</th><th>工具</th>"
            f"<th>Agent 回复</th><th>判定</th><th>Baseline 回复</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>多轮对话评估报告</title>
<style>
:root{{--brand:#2563eb;--bg:#f4f6fb;--line:#e5e9f2;--text:#1f2937;--muted:#6b7280}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);padding:32px 20px}}
.wrap{{max-width:1280px;margin:0 auto}}
h1{{font-size:24px;margin-bottom:6px}} .sub{{color:var(--muted);font-size:13px;margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:22px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 4px 16px rgba(31,41,55,.05)}}
.card h2{{font-size:15px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--line)}}
.card.baseline h2{{color:var(--muted)}}
.tablecard table,.card table{{width:100%;border-collapse:collapse;font-size:13px}}
.card td{{padding:7px 4px;border-bottom:1px solid var(--line)}}
h2.sec{{font-size:14px;margin:22px 0 8px;font-weight:500}}
.cap{{font-size:11px;font-weight:400;color:var(--brand);background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;padding:2px 8px;margin-left:6px}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(31,41,55,.05);margin-bottom:18px}}
.data-table th,.data-table td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
.data-table th{{background:#f8fafc;font-weight:600;color:var(--muted);font-size:12px}}
td.q{{max-width:330px;word-break:break-word;color:#374151;line-height:1.6}}
td.base{{color:#9ca3af;max-width:230px}}
td.m{{font-size:11px;color:#059669;white-space:nowrap}} td.m:not(:empty){{color:#b45309}}
tr:last-child td{{border-bottom:none}}
.note{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px 18px;font-size:13px;line-height:1.8;margin-bottom:22px}}
</style></head><body><div class="wrap">
<h1>多轮对话评估报告</h1>
<div class="sub">生成时间 {meta['time']} · 模型 {meta['model']} · 用例 {meta['cases']} 条 / {meta['turns']} 轮</div>

<div class="cards">
  <div class="card"><h2>Agent（工具 + 记忆 + 兜底）</h2><table>{metric_rows(agent)}</table></div>
  <div class="card baseline"><h2>Baseline（裸大模型，有上下文无工具）</h2><table>{metric_rows(baseline)}</table></div>
</div>

<div class="note">
<b>这份报告测的是单轮测不出的东西：</b><br>
• <b>上下文继承率</b>：用户已给过订单号后，追问时是否还会重复索要「请提供订单号」。重复索要 = 记忆失效。<br>
• <b>工具选择准确率</b>：该查订单时查了没有；没给订单号时是否乱查；改口说退款时是否切换到建单/转人工。<br>
• <b>全轮通过率</b>：整段对话每一轮都达标才算通过，任何一轮出错都不算。<br>
• Baseline 同样拥有对话历史（不是失忆），差距全部来自「没有工具查真实数据」。
</div>

{''.join(blocks)}
</div></body></html>"""


def _parse_num(v) -> float | None:
    """把 '90.0%' / '4198 ms' / '0' 解析成数值，用于跨次汇总"""
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else None


def aggregate(runs: list[dict]) -> dict:
    """多次运行的均值与极差 —— 单次结果方差过大，不能直接采信"""
    keys = ["事实命中率", "上下文继承率", "工具选择准确率", "禁止词命中次数",
            "全轮通过率", "平均单轮耗时"]
    out = {}
    for k in keys:
        vals = [x["agent"]["metrics"].get(k) for x in runs]
        nums = [n for n in (_parse_num(v) for v in vals) if n is not None]
        if not nums:
            continue
        unit = "%" if k.endswith("率") else (" ms" if "耗时" in k else "")
        out[k] = {
            "mean": f"{sum(nums)/len(nums):.1f}{unit}",
            "min": f"{min(nums):.1f}{unit}",
            "max": f"{max(nums):.1f}{unit}",
            "runs": len(nums),
        }
    # baseline 只关心事实与全轮通过
    for k in ["事实命中率", "全轮通过率"]:
        nums = [n for n in (_parse_num(x["baseline"]["metrics"].get(k)) for x in runs) if n is not None]
        if nums:
            out.setdefault("baseline_" + k, {
                "mean": f"{sum(nums)/len(nums):.1f}%", "min": f"{min(nums):.1f}%",
                "max": f"{max(nums):.1f}%", "runs": len(nums),
            })
    return out


def build_summary_html(agg: dict, repeat: int, meta: dict) -> str:
    order = ["事实命中率", "上下文继承率", "工具选择准确率", "禁止词命中次数",
             "全轮通过率", "平均单轮耗时", "baseline_事实命中率", "baseline_全轮通过率"]
    rows = []
    for k in order:
        if k not in agg:
            continue
        v = agg[k]
        name = k.replace("baseline_", "Baseline · ")
        rows.append(f"<tr><td>{name}</td><td class='hl'>{v['mean']}</td>"
                    f"<td>{v['min']}</td><td>{v['max']}</td><td>{v['runs']}</td></tr>")
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>多轮评估 · 多次运行汇总</title>
<style>
:root{{--bg:#f4f6fb;--line:#e5e9f2;--text:#1f2937;--muted:#6b7280}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);padding:32px 20px}}
.wrap{{max-width:820px;margin:0 auto}}
h1{{font-size:22px;margin-bottom:6px}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:22px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(31,41,55,.05)}}
th,td{{padding:10px 13px;text-align:left;border-bottom:1px solid var(--line)}}
th{{background:#f8fafc;font-size:12px;color:var(--muted);font-weight:600}}
td.hl{{font-weight:600;color:#185FA5}}
tr:last-child td{{border-bottom:none}}
.note{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px 18px;font-size:13px;line-height:1.8;margin-bottom:20px}}
</style></head><body><div class="wrap">
<h1>多轮评估 · {repeat} 次运行汇总</h1>
<div class="sub">{meta['time']} · 模型 {meta['model']} · {meta['cases']} 条用例 / {meta['turns']} 轮</div>
<div class="note">
<b>为什么要跑 {repeat} 次：</b>同一套用例两次单跑，全轮通过率曾出现 71.4% 与 42.9% 的差异（28 个百分点）。
temperature=0.1 仍有明显抖动，单次数字不可直接引用。下表为 {repeat} 次独立运行的均值与极差，
<b>结论以均值为主，极差反映稳定性</b>。
</div>
<table><thead><tr><th>指标</th><th>均值</th><th>最低</th><th>最高</th><th>运行次数</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""


async def main():
    repeat = 1
    if "--repeat" in sys.argv:
        repeat = max(1, int(sys.argv[sys.argv.index("--repeat") + 1]))

    cases = [json.loads(l) for l in MULTI_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"多轮用例: {len(cases)} 条 / {sum(len(c['turns']) for c in cases)} 轮 | 运行 {repeat} 次")

    all_runs = []
    meta = {}
    for rep in range(1, repeat + 1):
        # 每轮重置临时库，避免上一轮建的工单触发幂等逻辑、影响下一轮判定
        shutil.copy2(ROOT / "cs" / "data" / "cs.db", _TMP_DB)
        print(f"\n===== 第 {rep}/{repeat} 次 =====")

        t0 = time.time()
        agent_runs = [await run_agent_case(c) for c in cases]
        print(f"  Agent 用时 {time.time()-t0:.0f}s")

        client = AsyncOpenAI(api_key=CSConfig.API_KEY, base_url=CSConfig.BASE_URL)
        t1 = time.time()
        baseline_runs = [await run_baseline_case(c, client) for c in cases]
        await client.close()
        print(f"  Baseline 用时 {time.time()-t1:.0f}s")

        agent = summarize(cases, agent_runs, with_tools=True)
        baseline = summarize(cases, baseline_runs, with_tools=False)
        meta = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": CSConfig.MODEL,
                "cases": len(cases), "turns": sum(len(c["turns"]) for c in cases)}
        all_runs.append({"agent": agent, "baseline": baseline})

        print(f"  事实 {agent['metrics']['事实命中率']} | 继承 {agent['metrics']['上下文继承率']} "
              f"| 工具 {agent['metrics']['工具选择准确率']} | 全轮通过 {agent['metrics']['全轮通过率']}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {"meta": meta, "agent": agent, "baseline": baseline}
        (REPORT_DIR / f"multi_turn_{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (REPORT_DIR / f"multi_turn_{stamp}.html").write_text(
            build_html(agent, baseline, meta), encoding="utf-8")
        print(f"  报告: multi_turn_{stamp}.html")

    agg = aggregate(all_runs) if repeat > 1 else {}
    if repeat > 1:
        print("\n" + "=" * 62)
        print("  多次运行汇总（均值 / 最低 / 最高）")
        print("=" * 62)
        for k, v in agg.items():
            print(f"  {k.replace('baseline_','Baseline·'):24s} {v['mean']:>10s}  [{v['min']} ~ {v['max']}]")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spath = REPORT_DIR / f"multi_turn_summary_{stamp}.html"
        spath.write_text(build_summary_html(agg, repeat, meta), encoding="utf-8")
        (REPORT_DIR / f"multi_turn_summary_{stamp}.json").write_text(
            json.dumps({"repeat": repeat, "meta": meta, "aggregate": agg}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n汇总报告: {spath}")

    shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
