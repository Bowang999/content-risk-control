# -*- coding: utf-8 -*-
"""第 1 步：内容层基线（TF-IDF 字符 n-gram + 逻辑回归）。
同时完成：
  1) 从基线模型里提取“敏感字”权重；
  2) 用敏感字定向构造对抗测试集（data/adv_test.csv），后续所有模型都在同一份对抗集上评估；
  3) 对比 原始基线 vs 归一化 + 对抗增强 后的基线。
"""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import best_f1_threshold, binary_report, load_cold, save_json
from config import RESULTS, set_seed
from perturb import CJK, HAS_PINYIN, build_homophone_table, normalize
from text_utils import ADV_TEST_PATH, adv_eval, augment_train, build_adv_test


def fit_tfidf_lr(texts, labels):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True,
                          max_features=300000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=4.0, max_iter=2000)
    clf.fit(X, labels)
    return vec, clf


def score(vec, clf, texts):
    return clf.predict_proba(vec.transform(texts))[:, 1]


def main():
    set_seed()
    t0 = time.time()
    d = load_cold()
    tr, dv, te = d["train"], d["dev"], d["test"]
    print(f"COLD: train={len(tr)} dev={len(dv)} test={len(te)}  违规占比(train)={tr.label.mean():.3f}")
    if not HAS_PINYIN:
        print("[提示] 没装 pypinyin，同音字/拼音变体会被跳过：pip install pypinyin")

    # ---------- 1) 原始基线 ----------
    vec, clf = fit_tfidf_lr(tr.text, tr.label)
    th = best_f1_threshold(dv.label, score(vec, clf, dv.text))
    rep_raw = {"dev": binary_report(dv.label, score(vec, clf, dv.text), th),
               "test": binary_report(te.label, score(vec, clf, te.text), th)}
    print("基线 test:", {k: round(v, 4) for k, v in rep_raw["test"].items() if isinstance(v, float)})

    # ---------- 2) 敏感字权重 ----------
    vocab = vec.vocabulary_
    coef = clf.coef_[0]
    char_weight = {t: float(coef[j]) for t, j in vocab.items() if len(t) == 1 and CJK.match(t)}
    top = sorted(char_weight.items(), key=lambda x: -x[1])[:50]
    save_json(char_weight, RESULTS / "char_weight.json")
    print("权重最高的敏感字:", "".join(c for c, _ in top[:30]))

    # ---------- 3) 对抗测试集 ----------
    homo = build_homophone_table(tr.text.tolist())
    adv = build_adv_test(te, char_weight, homo)
    adv.to_csv(ADV_TEST_PATH, index=False, encoding="utf-8-sig")
    print(f"对抗测试集: {len(adv)} 条，按变体: {adv.method.value_counts().to_dict()}")
    print("示例:")
    for m, g in adv[adv.label == 1].groupby("method"):
        r = g.iloc[0]
        print(f"  [{m}] {r.text_orig[:40]}  ->  {r.text_adv[:40]}")

    adv_raw = adv_eval(adv, score(vec, clf, adv.text_orig), score(vec, clf, adv.text_adv), th)

    # ---------- 4) 防御：归一化 + 对抗增强 ----------
    aug, n_extra = augment_train(tr, char_weight, homo)
    vec2, clf2 = fit_tfidf_lr(aug.text.map(normalize), aug.label)
    s = lambda texts: score(vec2, clf2, [normalize(t) for t in texts])
    th2 = best_f1_threshold(dv.label, s(dv.text))
    rep_rob = {"dev": binary_report(dv.label, s(dv.text), th2),
               "test": binary_report(te.label, s(te.text), th2),
               "n_augmented": n_extra}
    adv_rob = adv_eval(adv, s(adv.text_orig), s(adv.text_adv), th2)

    res = {"baseline_raw": {**rep_raw, "adversarial": adv_raw},
           "baseline_robust": {**rep_rob, "adversarial": adv_rob},
           "top_sensitive_chars": top[:30],
           "adv_examples": adv[adv.label == 1].groupby("method").head(3)
                           [["method", "text_orig", "text_adv"]].to_dict("records"),
           "seconds": round(time.time() - t0, 1)}
    save_json(res, RESULTS / "s01_text_baseline.json")

    print("\n对抗评估（违规样本召回: 原文 -> 变体 | 绕过率）")
    for name, a in (("原始基线", adv_raw), ("归一化+对抗增强", adv_rob)):
        for m, v in a.items():
            if v["recall_orig"] is not None:
                print(f"  {name:10s} {m:14s} {v['recall_orig']:.3f} -> {v['recall_adv']:.3f} | "
                      f"绕过 {v['evasion_rate']:.3f} | 正常样本误判 {v['fpr_adv']}")
    print(f"完成，用时 {res['seconds']}s，结果: results/s01_text_baseline.json")


if __name__ == "__main__":
    main()
