"""
检索层对照实验 —— TF-IDF vs Embedding vs Hybrid

四组实验：
A. 同义改写鲁棒性：50 条 query（10 原始 + 40 口语化改写），测 Recall@1 / MRR@5
B. 规模扩展性：知识库从 31 条膨胀到 10031 条（注入近邻+跨域干扰），
   测 Recall@1 与 MRR@5 的退化，直接回答「1 万条怎么办」
C. 误召回率：7 条「库中无此主题」的问题，是否被误判为命中
   （关系到「未命中 -> 转人工」兜底是否还有效）
D. 成本：单次检索延迟、建索引耗时、依赖体积

实验与线上共用 cs/retrieval.py 的同一个 Retriever 类。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs.retrieval import Retriever, THRESHOLDS, FAQ_PATH  # noqa: E402

EVAL = Path(__file__).resolve().parent
REPORT_DIR = EVAL / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FAQS = json.loads(FAQ_PATH.read_text(encoding="utf-8"))["faqs"]
FAQ_DOCS = [f'{x["question"]} {x["answer"]}' for x in FAQS]

# 干扰语料是生成物（已 gitignore），缺失时自动重建，保证换台机器也能一键复现
DIST_PATH = EVAL / "distractors.json"
if not DIST_PATH.exists():
    from cs.eval.make_distractors import make
    make()
DISTRACTORS = json.loads(DIST_PATH.read_text(encoding="utf-8"))

BACKENDS = ("tfidf", "embedding", "hybrid")
SCALES = (0, 100, 500, 1000, 3000, 10000)
NEG_CATEGORIES = {"知识库未命中", "抗误召回", "越界拒答"}


def load_queries():
    """50 条带 target 的 query：10 原始 + 40 同义改写"""
    qs = []
    for line in (EVAL / "paraphrase.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        qs.append((f'{c["id"]}-ORIG', c["query"], c["target"], "原始"))
        for i, p in enumerate(c["paraphrases"], 1):
            qs.append((f'{c["id"]}-P{i}', p, c["target"], "改写"))
    return qs


def load_negatives():
    out = []
    for line in (EVAL / "testset.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c["category"] in NEG_CATEGORIES:
            out.append((c["id"], c["question"]))
    return out


def evaluate(r: Retriever, queries):
    """返回 (recall@1, recall@3, mrr@5, 平均延迟ms, 平均top1分数)

    线上 search_faq 默认返回 top3 交给 Agent 自行判断，因此 Recall@3
    比 Recall@1 更贴近真实业务；两者都报，避免用过于严格的 top1 低估系统。
    """
    c1, c3, rr, lat, top1_scores = 0, 0, 0.0, [], []
    for _, q, target, _ in queries:
        t0 = time.perf_counter()
        hits = r.search(q, k=5)
        lat.append((time.perf_counter() - t0) * 1000)
        top1_scores.append(hits[0][1])
        rank = next((i + 1 for i, (p, _) in enumerate(hits) if p["id"] == target), 0)
        if rank == 1:
            c1 += 1
        if target in [p["id"] for p, _ in hits[:3]]:
            c3 += 1
        if rank:
            rr += 1.0 / rank
    n = len(queries)
    return c1 / n, c3 / n, rr / n, float(np.mean(lat)), float(np.mean(top1_scores))


def exp_a(queries):
    out = {}
    for b in BACKENDS:
        r = Retriever(FAQ_DOCS, FAQS, b)
        by_kind = {}
        for kind in ("原始", "改写"):
            sub = [x for x in queries if x[3] == kind]
            r1, r3, mrr, lat, sc = evaluate(r, sub)
            by_kind[kind] = {"n": len(sub), "recall1": r1, "recall3": r3,
                             "mrr5": mrr, "top1_score": sc}
        r1, r3, mrr, lat, sc = evaluate(r, queries)
        out[b] = {"by_kind": by_kind, "all": {"n": len(queries), "recall1": r1,
                                              "recall3": r3, "mrr5": mrr,
                                              "latency_ms": lat}}
        print(f"  [{b:9s}] 原始 R@1 {by_kind['原始']['recall1']*100:5.1f}% / "
              f"R@3 {by_kind['原始']['recall3']*100:5.1f}% | "
              f"改写 R@1 {by_kind['改写']['recall1']*100:5.1f}% / "
              f"R@3 {by_kind['改写']['recall3']*100:5.1f}% | "
              f"MRR@5 {mrr:.3f} | 单次 {lat:.2f}ms")
    return out


def exp_bcd(queries, negatives):
    rows = []
    print("\n规模实验（每档位: 后端 / Recall@1 / MRR@5 / 误召回率 / 单次延迟 / 建索引）")
    for n in SCALES:
        payloads = FAQS + DISTRACTORS[:n]
        docs = [f'{x["question"]} {x["answer"]}' for x in payloads]
        for b in BACKENDS:
            t0 = time.perf_counter()
            r = Retriever(docs, payloads, b)
            build_ms = (time.perf_counter() - t0) * 1000
            r.search(queries[0][1], k=5)  # 预热

            r1, r3, mrr, lat, sc = evaluate(r, queries)
            # 误召回率只在原始 31 条库上有意义：注入干扰项后，部分负例主题
            # （如「团建经费」）本就出现在干扰语料里，已不再是负例。
            neg_scores = [r.search(q, k=1)[0][1] for _, q in negatives] if n == 0 else []
            false_hit = (float(np.mean([s >= r.threshold for s in neg_scores]))
                         if neg_scores else None)
            row = {"scale": n, "docs": len(docs), "backend": b,
                   "recall1": r1, "recall3": r3, "mrr5": mrr, "false_hit": false_hit,
                   "neg_mean": float(np.mean(neg_scores)) if neg_scores else None,
                   "latency_ms": lat, "build_ms": build_ms, "threshold": r.threshold}
            rows.append(row)
            fh = f"{false_hit*100:4.1f}%" if false_hit is not None else "  n/a"
            print(f"  {n:5d} 条 | {b:9s} | R@1 {r1*100:5.1f}% | R@3 {r3*100:5.1f}% | "
                  f"MRR {mrr:.3f} | 误召回 {fh} | {lat:6.2f}ms | 建索引 {build_ms/1000:5.1f}s")
    return rows


def build_html(a, rows, calib, neg_n):
    def pct(x):
        return f"{x*100:.1f}%"

    # 同义改写表
    a_rows = "".join(
        f"<tr><td><b>{b}</b></td>"
        f"<td>{pct(v['by_kind']['原始']['recall1'])}</td><td>{pct(v['by_kind']['原始']['recall3'])}</td>"
        f"<td>{pct(v['by_kind']['改写']['recall1'])}</td><td>{pct(v['by_kind']['改写']['recall3'])}</td>"
        f"<td>{v['all']['mrr5']:.3f}</td><td>{v['all']['latency_ms']:.2f} ms</td></tr>"
        for b, v in a.items()
    )

    # 规模表：按档位分组
    tables = []
    for n in SCALES:
        sub = [r for r in rows if r["scale"] == n]
        tr = "".join(
            f"<tr><td><b>{r['backend']}</b></td><td>{pct(r['recall1'])}</td>"
            f"<td>{pct(r['recall3'])}</td><td>{r['mrr5']:.3f}</td>"
            f"<td>{pct(r['false_hit']) if r['false_hit'] is not None else '—'}</td>"
            f"<td>{r['latency_ms']:.2f}</td><td>{r['build_ms']/1000:.1f} s</td></tr>"
            for r in sub
        )
        tables.append(
            f"<h3>知识库 {n + 31} 条（31 条真 FAQ + {n} 条干扰）</h3>"
            f"<table><thead><tr><th>后端</th><th>Recall@1</th><th>Recall@3</th><th>MRR@5</th>"
            f"<th>误召回率</th><th>单次检索</th><th>建索引</th></tr></thead><tbody>{tr}</tbody></table>"
        )

    # 规模退化曲线（SVG 折线）
    colors = {"tfidf": "#2563eb", "embedding": "#dc2626", "hybrid": "#059669"}
    W, H, PAD = 620, 240, 44

    def xy(n, v):
        i = list(SCALES).index(n)
        x = PAD + i * (W - 2 * PAD) / (len(SCALES) - 1)
        y = H - PAD - (v / 100.0) * (H - 2 * PAD)
        return x, y

    paths, dots = [], []
    for b in BACKENDS:
        pts = [(n, next(r["recall1"] for r in rows if r["scale"] == n and r["backend"] == b) * 100)
               for n in SCALES]
        d = " ".join(f"{'M' if i == 0 else 'L'}{xy(n, v)[0]:.1f},{xy(n, v)[1]:.1f}"
                     for i, (n, v) in enumerate(pts))
        paths.append(f'<path d="{d}" fill="none" stroke="{colors[b]}" stroke-width="2.5"/>')
        for n, v in pts:
            x, y = xy(n, v)
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[b]}"/>')
            dots.append(f'<text x="{x:.1f}" y="{y-9:.1f}" font-size="10" fill="{colors[b]}" '
                        f'text-anchor="middle">{v:.0f}</text>')
    axis = (
        f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#cbd5e1"/>'
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#cbd5e1"/>'
    )
    for i, n in enumerate(SCALES):
        x = PAD + i * (W - 2 * PAD) / (len(SCALES) - 1)
        dots.append(f'<text x="{x:.1f}" y="{H-PAD+16}" font-size="10" fill="#64748b" '
                    f'text-anchor="middle">{n+31}</text>')
    for v in (0, 50, 100):
        y = H - PAD - v / 100.0 * (H - 2 * PAD)
        dots.append(f'<text x="{PAD-8}" y="{y+3}" font-size="10" fill="#64748b" '
                    f'text-anchor="end">{v}%</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:660px">'
           f'{axis}{"".join(paths)}{"".join(dots)}</svg>')

    legend = " ".join(
        f'<span style="margin-right:18px"><i style="display:inline-block;width:12px;height:3px;'
        f'background:{c};vertical-align:middle"></i> {b}</span>'
        for b, c in colors.items()
    )

    cal_rows = "".join(
        f"<tr><td><b>{b}</b></td><td>{v['mean_pos']:.3f}</td><td>{v['mean_neg']:.3f}</td>"
        f"<td>{'是' if v['separable'] else '否'}</td><td>{pct(v['best_acc'])}</td>"
        f"<td>{v['current']}</td><td>{pct(v.get('current_acc', 0))}</td></tr>"
        for b, v in calib.items()
    )

    def pick(n, b):
        return next(r for r in rows if r["scale"] == n and r["backend"] == b)

    tf0, em0 = pick(0, "tfidf"), pick(0, "embedding")
    tf_end, em_end = pick(10000, "tfidf"), pick(10000, "embedding")
    hy_end = pick(10000, "hybrid")

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>检索层对照实验报告</title>
<style>
body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#f7f9fc;color:#1e293b;
margin:0;padding:32px;line-height:1.7}}
.wrap{{max-width:960px;margin:0 auto;background:#fff;border:1px solid #e2e8f0;
border-radius:14px;padding:32px 40px;box-shadow:0 2px 12px rgba(0,0,0,.04)}}
h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:18px;margin:32px 0 12px;
border-left:4px solid #2563eb;padding-left:10px}} h3{{font-size:14px;margin:20px 0 8px;color:#475569}}
.sub{{color:#64748b;font-size:13px;margin-bottom:24px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:10px}}
th,td{{border:1px solid #e2e8f0;padding:8px 10px;text-align:left}}
th{{background:#f1f5f9;font-weight:600}} tr:nth-child(even) td{{background:#fafcff}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.card{{flex:1;min-width:180px;background:#f8fafc;border:1px solid #e2e8f0;
border-radius:10px;padding:14px 16px}}
.card .n{{font-size:24px;font-weight:700;color:#2563eb}}
.card .t{{font-size:12px;color:#64748b}}
.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;
font-size:13px;border-radius:6px;margin:14px 0}}
.good{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px 16px;
font-size:13px;border-radius:6px;margin:14px 0}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>检索层对照实验报告</h1>
<div class="sub">生成时间 {datetime.now():%Y-%m-%d %H:%M:%S} · 模型 BAAI/bge-small-zh-v1.5 (ONNX, 512维) ·
语料 31 条真 FAQ + 最多 10000 条干扰 · 50 条标注 query + {neg_n} 条负例</div>

<div class="kpi">
<div class="card"><div class="n">{pct(a['embedding']['by_kind']['改写']['recall1'])}</div>
<div class="t">Embedding 同义改写 Recall@1（TF-IDF 为 {pct(a['tfidf']['by_kind']['改写']['recall1'])}）</div></div>
<div class="card"><div class="n">{pct(em_end['recall1'])}</div>
<div class="t">Embedding 在 10031 条时 Recall@1（TF-IDF 为 {pct(tf_end['recall1'])}）</div></div>
<div class="card"><div class="n">{em_end['latency_ms']:.2f} ms</div>
<div class="t">1 万条语料下 Embedding 单次检索延迟</div></div>
</div>

<h2>一、同义改写鲁棒性</h2>
<p>10 条真实政策问题 + 40 条口语化改写（如「怎么退货」→「不想要了能退吗」）。
TF-IDF 依赖词面重叠，改写后大量 token 无法匹配，这是它最致命的短板。
<b>同时报告 Recall@1 与 Recall@3</b>：线上 <code>search_faq</code> 默认返回 top3
交给 Agent 自行判断，因此 R@3 比只看第一名的 R@1 更贴近真实业务。
原始问法本身也存在真实歧义（例如「能开发票吗」在 F012「怎么开发票」与
F013「发票可以改开」之间只差 0.001），并非实现 bug。</p>
<table><thead><tr><th>后端</th><th>原始 R@1</th><th>原始 R@3</th><th>改写 R@1</th><th>改写 R@3</th>
<th>整体 MRR@5</th><th>单次检索延迟</th></tr></thead><tbody>{a_rows}</tbody></table>

<h2>二、阈值校准（命中 / 未命中是否可分）</h2>
<p>正例 50 条（政策咨询 + 改写），负例 {neg_n} 条「知识库中根本没有该主题」的问题。
第一版校准误把「幻觉诱导」类纳入负例（例如「账号被盗了，少 2000 块，你们补给我」其实命中
F030 是合理的），修正后才得到下表。<b>三者都不完全可分</b>，说明单靠阈值无法完美判定，
这也是 Agent 保留「检索命中但内容不相关不得硬答」prompt 规则的原因。</p>
<table><thead><tr><th>后端</th><th>正例均分</th><th>负例均分</th><th>区间可分</th>
<th>最优准确率</th><th>采用阈值</th><th>该阈值下实际准确率</th></tr></thead><tbody>{cal_rows}</tbody></table>
<div class="note">「最优准确率」是扫描全阈值得到的上限，<b>不等于</b>线上实际表现；
最后一列才是采用阈值下的真实准确率。<br>
TF-IDF 保持 0.15（线上已验证值）而不取最优 0.055：后者会显著抬高误召回，
且 0.15 是在<b>标准问法</b>上标定的；本表正例含 40 条口语化改写，
TF-IDF 对它们大量给出 0 分，故实际准确率只有 82.5% —— 这恰好量化了
「TF-IDF 不适合改写场景」，也是本次补 Embedding 的直接动机。<br>
Hybrid 的最优区间仅 0.407~0.422，明显偏窄，存在对小标注集过拟合的风险，
已在代码注释中标明，后续应扩充标注集复验。</div>

<h2>三、规模扩展性：知识库膨胀到 1 万条</h2>
<p>干扰项按真实知识库膨胀形态构造：同域近邻（品类/渠道/时效替换，措辞高度相似）+ 跨域条目。
「正确条目」始终只有原始 31 条 FAQ，因此 Recall 标注无歧义。</p>
<div class="note"><b>误召回率只在 31 条原始库（0 干扰）下报告</b>：注入干扰项后，
部分负例主题（如「公司团建经费标准」）本身就出现在跨域干扰语料里，
此时该问题已不再是「知识库无此主题」的负例，指标跨规模不可比，故标为「—」。</div>
<div>{legend}</div>{svg}
<p style="font-size:12px;color:#64748b">横轴：知识库总条数；纵轴：Recall@1。曲线越平，规模鲁棒性越好。</p>
{"".join(tables)}

<div class="good"><b>结论</b>：知识库从 31 条膨胀到 10031 条，TF-IDF 的 Recall@1 从
{pct(tf0['recall1'])} 降到 {pct(tf_end['recall1'])}，
Embedding 从 {pct(em0['recall1'])} 仅降到 {pct(em_end['recall1'])}；
1 万条时 Recall@3 分别为 {pct(tf_end['recall3'])} / {pct(em_end['recall3'])} /
{pct(hy_end['recall3'])}（TF-IDF / Embedding / Hybrid）。
单次检索延迟 {tf_end['latency_ms']:.2f} / {em_end['latency_ms']:.2f} /
{hy_end['latency_ms']:.2f} ms。建索引是一次性离线成本（Embedding 1 万条约
{em_end['build_ms']/1000:.0f} s），不计入在线延迟。Hybrid 采用 RRF 排名融合，
是否优于单后端以实测为准，不做预设。</div>

<h2>四、成本与工程约束</h2>
<table><thead><tr><th>项</th><th>TF-IDF</th><th>Embedding</th></tr></thead><tbody>
<tr><td>额外依赖</td><td>scikit-learn + jieba（已有）</td><td>fastembed + ONNX Runtime（无 torch）</td></tr>
<tr><td>模型文件</td><td>无</td><td>约 90 MB，缓存在项目 .model_cache/</td></tr>
<tr><td>首次加载</td><td>&lt; 0.1 s</td><td>约 19 s（含下载），之后从缓存秒级</td></tr>
<tr><td>是否需联网</td><td>否</td><td>仅首次下载模型，之后完全离线</td></tr>
</tbody></table>

<h2>五、方法与局限（如实披露）</h2>
<div class="note">
1. <b>阈值并非完全可分</b>：Embedding 正例最低 {calib['embedding']['min_pos']:.3f}、
负例最高 {calib['embedding']['max_neg']:.3f}，存在重叠（主要是「能不能货到付款」误召回
F018「什么时候补货」）。阈值 0.50 是准确率最优折中，不是完美分割面。<br>
2. <b>干扰项是合成数据</b>：按真实膨胀形态构造，但不是真实业务语料，结论用于架构选型，
不等于线上真实分布下的效果。<br>
3. <b>语料规模上限 10031 条</b>：未测向量库（FAISS / Milvus）与 ANN 近似检索；
当前是 numpy 全量点积，1 万条仍可接受，更大规模应换 ANN 索引。<br>
4. <b>检索指标 ≠ 端到端指标</b>：最终是否提升客服回答质量，以 Agent 单轮/多轮评估为准。<br>
5. <b>Hybrid 方案经过一次返工</b>：最初用「各后端分数 / 各自阈值后加权求和」，
TF-IDF 被放大 6.7 倍而 Embedding 只放大 2 倍，导致 TF-IDF 主导融合，
Hybrid 改写 Recall@1 反而只有 45%（纯 Embedding 为 77.5%）。改用 RRF 排名融合后重测，
说明「融合前必须处理尺度，否则不如不融合」。<br>
6. <b>Recall@1 偏严格</b>：同主题 FAQ 之间存在真实歧义（F012/F013 仅差 0.001），
线上返回 top3 由 Agent 决策，故以 Recall@3 作为更贴近业务的指标一并报告。
</div>

<h2>复现命令</h2>
<p><code>python -m cs.eval.make_distractors</code> →
<code>python -m cs.eval.calibrate</code> →
<code>python -m cs.eval.run_retrieval_eval</code><br>
切换后端：<code>set CS_RETRIEVER=embedding</code>（Windows）或
<code>export CS_RETRIEVER=hybrid</code></p>
</div></body></html>"""


def main():
    queries = load_queries()
    negatives = load_negatives()
    print(f"query {len(queries)} 条 | 负例 {len(negatives)} 条 | 干扰池 {len(DISTRACTORS)} 条\n")

    print("=== 实验 A：同义改写鲁棒性 ===")
    a = exp_a(queries)
    print("\n=== 实验 B/C/D：规模扩展 + 误召回 + 成本 ===")
    rows = exp_bcd(queries, negatives)

    calib = json.loads((EVAL / "calibration.json").read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"meta": {"time": stamp, "faqs": len(FAQS), "queries": len(queries),
                        "negatives": len(negatives), "scales": list(SCALES),
                        "thresholds": THRESHOLDS},
               "paraphrase": a, "scale": rows, "calibration": calib}
    jp = REPORT_DIR / f"retrieval_{stamp}.json"
    hp = REPORT_DIR / f"retrieval_{stamp}.html"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hp.write_text(build_html(a, rows, calib, len(negatives)), encoding="utf-8")
    print(f"\nJSON: {jp}\nHTML: {hp}")


if __name__ == "__main__":
    main()
