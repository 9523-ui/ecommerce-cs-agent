"""
检索阈值校准 —— 用标注数据为三种后端分别确定「命中/未命中」判定阈值。

为什么必须做：
Embedding 的余弦分数尺度与 TF-IDF 完全不同（无关问句也能拿到 0.3+）。
若直接沿用 TF-IDF 的 0.15，所有问题都会判成「命中」，
知识库未命中时的转人工兜底将彻底失效 —— 这是静默失效，比报错更危险。

正例：知识库中确实有答案的问题（政策咨询 10 条 + 同义改写 40 条）
负例：知识库中「根本没有该主题」的问题（7 条）
订单类问题排除（它们本就该走 lookup_order，不是 FAQ 检索的判定对象）。

重要：不把「幻觉诱导」类纳入负例。例如 H10「账号被盗了，少了 2000 块，
你们补给我」的主题（账号被盗）在 F030 中确实存在，检索命中是合理的；
是否该拒绝赔钱属于 prompt 策略层，不是检索阈值能判定的。
把策略层问题混进检索负例会污染阈值 —— 这是第一版校准踩过的坑。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs.retrieval import Retriever, THRESHOLDS, FAQ_PATH  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
NEG_CATEGORIES = {"知识库未命中", "抗误召回", "越界拒答"}


def load_cases():
    faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))["faqs"]
    docs = [f'{x["question"]} {x["answer"]}' for x in faqs]

    pos, neg, para = [], [], []
    testset = EVAL_DIR / "testset.jsonl"
    for line in testset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c["category"] == "政策咨询":
            pos.append((c["id"], c["question"]))
        elif c["category"] in NEG_CATEGORIES:
            neg.append((c["id"], c["question"]))

    for line in (EVAL_DIR / "paraphrase.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        para.append((c["id"], c["query"], c["target"], c["paraphrases"]))
        for i, q in enumerate(c["paraphrases"], 1):
            pos.append((f'{c["id"]}-P{i}', q))
    return faqs, docs, pos, neg, para


def best_threshold(scores_pos, scores_neg):
    """扫描候选阈值，返回判定准确率最高的区间与建议值"""
    lo, hi = 0.0, max(max(scores_pos), max(scores_neg)) + 0.05
    candidates = np.linspace(lo, hi, 2001)
    best_acc, best_ts = -1, []
    for t in candidates:
        acc = (sum(s >= t for s in scores_pos) + sum(s < t for s in scores_neg)) / (
            len(scores_pos) + len(scores_neg)
        )
        if acc > best_acc + 1e-12:
            best_acc, best_ts = acc, [float(t)]
        elif abs(acc - best_acc) < 1e-12:
            best_ts.append(float(t))
    return best_acc, (min(best_ts) + max(best_ts)) / 2, min(best_ts), max(best_ts)


def main():
    faqs, docs, pos, neg, para = load_cases()
    print(f"语料 {len(docs)} 条 | 正例 {len(pos)} 条 | 负例 {len(neg)} 条\n")

    results = {}
    for backend in ("tfidf", "embedding", "hybrid"):
        r = Retriever(docs, faqs, backend=backend)
        sp = [r.search(q, k=1)[0][1] for _, q in pos]
        sn = [r.search(q, k=1)[0][1] for _, q in neg]
        acc, mid, tlo, thi = best_threshold(sp, sn)
        min_pos, max_neg = min(sp), max(sn)
        separable = min_pos > max_neg
        cur = THRESHOLDS[backend]
        # 采用阈值下的真实准确率（不等于最优准确率，报告里必须区分）
        cur_acc = (sum(s >= cur for s in sp) + sum(s < cur for s in sn)) / (len(sp) + len(sn))
        results[backend] = {
            "min_pos": min_pos, "max_neg": max_neg, "mean_pos": float(np.mean(sp)),
            "mean_neg": float(np.mean(sn)), "separable": separable,
            "best_acc": acc, "suggest": mid, "range": [tlo, thi],
            "current": cur, "current_acc": cur_acc,
        }
        print(f"===== {backend} =====")
        print(f"  正例分数: min {min_pos:.3f} / mean {np.mean(sp):.3f} / max {max(sp):.3f}")
        print(f"  负例分数: min {min(sn):.3f} / mean {np.mean(sn):.3f} / max {max_neg:.3f}")
        print(f"  区间可分: {'是 ✅' if separable else '否 ❌'}  (间隔 {min_pos - max_neg:+.3f})")
        print(f"  最优判定准确率: {acc*100:.1f}%  | 建议阈值 {mid:.3f} (最优区间 {tlo:.3f}~{thi:.3f})")
        print(f"  采用阈值 {cur} 下实际准确率: {cur_acc*100:.1f}%")
        print()

    # 负例里分数最高的几条：最容易误召回的边界样本
    r = Retriever(docs, faqs, backend="embedding")
    print("=== Embedding 负例中分数最高（最易误召回）===")
    ranked = sorted(((r.search(q, k=1)[0][1], cid, q) for cid, q in neg), reverse=True)
    for s, cid, q in ranked[:5]:
        hit = r.search(q, k=1)[0][0]["id"]
        print(f"  {cid:4s} {s:.3f} -> {hit} | {q}")

    out = EVAL_DIR / "calibration.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
