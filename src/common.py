# -*- coding: utf-8 -*-
"""数据读取与通用评估指标。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import COLD_DIR

TEXT_CANDIDATES = ["TEXT", "text", "Text", "content", "sentence", "comment"]
LABEL_CANDIDATES = ["label", "Label", "labels"]


def _pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"找不到{what}列，现有列：{list(cols)}")


def _find_csv(name: str) -> Path:
    hits = sorted(COLD_DIR.rglob(name))
    if not hits:
        raise FileNotFoundError(
            f"在 {COLD_DIR} 下找不到 {name}。请把 COLDataset 里的 train.csv/dev.csv/test.csv 放进 data/COLD/")
    return hits[0]


def load_cold():
    """返回 {'train': df, 'dev': df, 'test': df}，统一为 text / label 两列（label: 1=冒犯/违规, 0=安全）。"""
    out = {}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(_find_csv(f"{split}.csv"))
        tcol = _pick(df.columns, TEXT_CANDIDATES, "文本")
        lcol = _pick(df.columns, LABEL_CANDIDATES, "标签")
        d = pd.DataFrame({"text": df[tcol].astype(str).str.strip(), "label": df[lcol].astype(int)})
        if "topic" in df.columns:
            d["topic"] = df["topic"].astype(str)
        d = d[d["text"].str.len() > 0].reset_index(drop=True)
        out[split] = d
    return out


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def binary_report(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    rep = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
    }
    if len(np.unique(y)) == 2:
        rep["auc"] = float(roc_auc_score(y, p))
        rep["pr_auc"] = float(average_precision_score(y, p))
        rep["ks"] = ks_stat(y, p)
    return rep


def recall_at_alert_rate(y, p, rates=(0.01, 0.02, 0.05)):
    """按分数从高到低只审核前 rate 比例的样本，能抓到多少比例的正样本。"""
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(p))
    out = {}
    for r in rates:
        k = max(1, int(round(len(y) * r)))
        top = y[order[:k]]
        out[f"top{int(r*100)}%"] = {"recall": float(top.sum() / max(1, y.sum())),
                                    "precision": float(top.mean())}
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
