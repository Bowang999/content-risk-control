# -*- coding: utf-8 -*-
"""内容层共用：对抗测试集构造、训练集对抗增强、对抗评估。"""
import random

import numpy as np
import pandas as pd

from config import DATA, SEED
from perturb import METHODS, perturb

ADV_TEST_PATH = DATA / "adv_test.csv"


def build_adv_test(test_df, char_weight, homo_table, ratio=0.2, seed=SEED):
    """对测试集每条样本 × 每种变体做一次改写，只保留确实发生改写的样本。"""
    rng = random.Random(seed)
    rows = []
    for i, r in test_df.iterrows():
        for m in METHODS:
            adv = perturb(r.text, m, char_weight, homo_table, ratio, rng)
            if adv != r.text:
                rows.append({"idx": i, "method": m, "label": int(r.label),
                             "text_orig": r.text, "text_adv": adv})
    return pd.DataFrame(rows)


def augment_train(train_df, char_weight, homo_table, frac=0.5, ratio=0.2, seed=SEED + 1):
    """对抗增强：抽取 frac 比例的违规样本 + 等量正常样本做随机变体，追加进训练集。
    正常样本也加扰动，避免模型学成“有怪符号就判违规”。"""
    rng = random.Random(seed)
    pos = train_df[train_df.label == 1].sample(frac=frac, random_state=seed)
    neg = train_df[train_df.label == 0].sample(n=min(len(pos), (train_df.label == 0).sum()),
                                               random_state=seed)
    extra = []
    for _, r in pd.concat([pos, neg]).iterrows():
        m = rng.choice(METHODS)
        adv = perturb(r.text, m, char_weight, homo_table, ratio, rng)
        if adv != r.text:
            extra.append({"text": adv, "label": int(r.label)})
    aug = pd.concat([train_df[["text", "label"]], pd.DataFrame(extra)], ignore_index=True)
    return aug.sample(frac=1.0, random_state=seed).reset_index(drop=True), len(extra)


def adv_eval(adv_df, p_orig, p_adv, threshold):
    """p_orig / p_adv：与 adv_df 行对齐的原文分数和变体分数。
    违规样本看召回（越高越好，1-召回 = 绕过率），正常样本看误判率。"""
    d = adv_df.copy()
    d["hit_orig"] = (np.asarray(p_orig) >= threshold).astype(int)
    d["hit_adv"] = (np.asarray(p_adv) >= threshold).astype(int)
    out = {}
    for m, g in list(d.groupby("method")) + [("ALL", d)]:
        pos, neg = g[g.label == 1], g[g.label == 0]
        out[m] = {
            "n_offensive": int(len(pos)),
            "recall_orig": float(pos.hit_orig.mean()) if len(pos) else None,
            "recall_adv": float(pos.hit_adv.mean()) if len(pos) else None,
            # 原文能抓到、改写后漏掉的比例：真正的“被绕过率”
            "evasion_rate": float(((pos.hit_orig == 1) & (pos.hit_adv == 0)).sum()
                                  / max(1, (pos.hit_orig == 1).sum())) if len(pos) else None,
            "n_safe": int(len(neg)),
            "fpr_orig": float(neg.hit_orig.mean()) if len(neg) else None,
            "fpr_adv": float(neg.hit_adv.mean()) if len(neg) else None,
        }
    return out
