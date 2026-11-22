# -*- coding: utf-8 -*-
"""第 5 步：决策层——模型分数分三档：自动处置 / 送人审 / 放行。

阈值都在验证集上选，测试集只用来报告：
  t_high：自动处置的精确率 >= 目标（控制误杀）——取满足条件的最低分数线；
  t_low ：分数 >= t_low 的样本覆盖 >= 目标召回——取满足条件的最高分数线；
  中间段送人审（假设人审判断正确）。
对比基准：全部人审（人审量 100%）。
"""
import numpy as np

from common import load_json, save_json
from config import DECISION_TARGETS, RESULTS


def pick_thresholds(y, p, auto_precision, pass_recall):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec = tp / k
    rec = tp / max(1, ys.sum())
    ok = np.where((prec >= auto_precision) & (k >= 30))[0]  # 至少 30 条，避免头部几条样本偶然全对
    # 取精确率仍达标的最深位置（自动处置量最大）
    t_high = float(ps[ok.max()]) if len(ok) else float(ps[0]) + 1e-9
    j = np.argmax(rec >= pass_recall)
    t_low = float(ps[j])
    t_low = min(t_low, t_high)
    return t_high, t_low


def route_report(y, p, t_high, t_low):
    y = np.asarray(y).astype(int)
    auto = p >= t_high
    review = (p >= t_low) & ~auto
    passed = p < t_low
    n, npos = len(y), max(1, y.sum())
    return {
        "auto_share": float(auto.mean()),
        "auto_precision": float(y[auto].mean()) if auto.any() else None,
        "auto_recall": float(y[auto].sum() / npos),
        "review_share": float(review.mean()),
        "review_pos_rate": float(y[review].mean()) if review.any() else None,
        "pass_share": float(passed.mean()),
        "final_recall": float(y[auto | review].sum() / npos),
        "missed_in_pass": int(y[passed].sum()),
        "wrongly_auto_disposed_share_of_safe": float((auto & (y == 0)).sum() / max(1, (y == 0).sum())),
        "review_volume_saved_vs_full_review": float(1 - review.mean()),
    }


def main():
    res = {}
    cands = [("content", "probs_bert_robust.npz"), ("content", "probs_bert_raw.npz"),
             ("fraud", "probs_fraud.npz")]
    done = set()
    for kind, fn in cands:
        path = RESULTS / fn
        if kind in done or not path.exists():
            continue
        z = np.load(path)
        dev, dev_y, te, te_y = z["dev"], z["dev_y"].astype(int), z["test"], z["test_y"].astype(int)
        tg = DECISION_TARGETS[kind]
        th, tl = pick_thresholds(dev_y, dev, tg["auto_precision"], tg["pass_recall"])
        rep = {"source": fn, "targets": tg, "t_high": th, "t_low": tl,
               "valid": route_report(dev_y, dev, th, tl), "test": route_report(te_y, te, th, tl)}
        # 敏感性：不同目标下的人审量
        sens = []
        for ap_ in (0.90, 0.95, 0.98) if kind == "content" else (0.6, 0.7, 0.8, 0.9):
            for pr in (0.90, 0.95, 0.98) if kind == "content" else (0.7, 0.8, 0.9):
                a, b = pick_thresholds(dev_y, dev, ap_, pr)
                r = route_report(te_y, te, a, b)
                sens.append({"auto_precision_target": ap_, "recall_target": pr,
                             **{k: r[k] for k in ("auto_share", "auto_precision", "review_share",
                                                  "final_recall")}})
        rep["sensitivity"] = sens
        # 验证集与测试集分布不一致时（COLD 的 dev 与 train 同源，test 是另外标注的），
        # 用测试集随机一半做阈值校准、另一半评估——相当于上线前用少量线上标注样本重新校准阈值
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(te_y))
        cal, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
        ch, cl = pick_thresholds(te_y[cal], te[cal], tg["auto_precision"], tg["pass_recall"])
        rep["recalibrated_on_half_test"] = {"t_high": ch, "t_low": cl,
                                            "calib_half": route_report(te_y[cal], te[cal], ch, cl),
                                            "eval_half": route_report(te_y[ev], te[ev], ch, cl)}
        res[kind] = rep
        done.add(kind)
        t = rep["test"]
        print(f"[{kind}] 来源 {fn}  t_high={th:.3f} t_low={tl:.3f}")
        print(f"   测试集：自动处置 {t['auto_share']:.1%}（精确率 {t['auto_precision']}），"
              f"人审 {t['review_share']:.1%}，放行 {t['pass_share']:.1%}，最终召回 {t['final_recall']:.1%}，"
              f"人审量比全量人审减少 {t['review_volume_saved_vs_full_review']:.1%}")
        e = rep["recalibrated_on_half_test"]["eval_half"]
        print(f"   用一半测试集重新校准阈值后（另一半评估）：自动处置 {e['auto_share']:.1%}（精确率 {e['auto_precision']}），"
              f"人审 {e['review_share']:.1%}，最终召回 {e['final_recall']:.1%}")

    # 大模型 vs 微调模型（同一批抽样）
    llm_path = RESULTS / "s03_llm.json"
    if llm_path.exists() and "content" in res:
        llm = load_json(llm_path)
        z = np.load(RESULTS / res["content"]["source"])
        idx = np.array(llm["sample_idx"])
        from common import binary_report
        res["llm_vs_bert_same_sample"] = {
            "llm": llm["clean_sample"],
            "bert": binary_report(z["test_y"][idx], z["test"][idx], float(z["threshold"]))}
    save_json(res, RESULTS / "s05_decision.json")
    print("完成，结果: results/s05_decision.json")


if __name__ == "__main__":
    main()
