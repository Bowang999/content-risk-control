# -*- coding: utf-8 -*-
"""第 4 步：账号/交易层反欺诈（Kaggle IEEE-CIS Fraud Detection）。

要点：
  * 按时间切分（前 80% 训练、后 20% 测试），模拟上线后对未来数据的预测；
  * 用 card1 + addr1 + 首次交易日 还原“用户 ID”（uid），构造用户历史行为特征；
  * 所有聚合特征只用“当前时刻及以前”的数据（累计统计），避免用到未来信息（穿越）；
  * 设备关联特征：一个设备指纹累计关联过多少个不同用户（团伙/共用设备信号）；
  * 消融实验：原始特征 → +用户行为特征 → +设备关联特征。
用法： python s04_fraud_lgb.py   [--use_v 额外加入 339 个 V 列]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import binary_report, recall_at_alert_rate, save_json
from config import IEEE_DIR, RESULTS, SEED, set_seed

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
BASE = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
         "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1",
         "P_emaildomain", "R_emaildomain"] + C_COLS + D_COLS + M_COLS)
ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]


def load(use_v):
    tr_path = next(IEEE_DIR.rglob("train_transaction.csv"), None)
    id_path = next(IEEE_DIR.rglob("train_identity.csv"), None)
    if tr_path is None:
        raise FileNotFoundError("请把 Kaggle IEEE-CIS 的 train_transaction.csv 和 train_identity.csv 放进 data/ieee-fraud/")
    cols = BASE + ([f"V{i}" for i in range(1, 340)] if use_v else [])
    df = pd.read_csv(tr_path, usecols=lambda c: c in cols)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    if id_path is not None:
        idf = pd.read_csv(id_path)
        idf.columns = [c.replace("-", "_") for c in idf.columns]
        df = df.merge(idf[[c for c in ID_COLS if c in idf.columns]], on="TransactionID", how="left")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def cum_distinct(df, key, val):
    """截至当前行，key 累计关联过多少个不同的 val。"""
    first = ~df.duplicated([key, val])
    return first.groupby(df[key]).cumsum()


def window_stats(key, t, amt, w):
    """同一 key 在 [t-w, t] 时间窗内的交易笔数和金额合计（只看过去，含当前笔）。"""
    codes = pd.factorize(key)[0].astype(np.int64)
    k = codes * 10**9 + t.astype(np.int64)          # 先按 key 再按时间排序的复合键
    order = np.argsort(k, kind="stable")
    ks = k[order]
    cum = np.concatenate([[0.0], np.cumsum(amt[order])])
    hi = np.searchsorted(ks, ks, side="right")
    lo = np.searchsorted(ks, ks - w, side="left")
    cnt, s = np.empty(len(k)), np.empty(len(k))
    cnt[order] = hi - lo
    s[order] = cum[hi] - cum[lo]
    return cnt.astype("float32"), s.astype("float32")


def build_features(df):
    day = df.TransactionDT / 86400
    df["hour"] = ((df.TransactionDT / 3600) % 24).astype("float32")
    df["weekday"] = (np.floor(day) % 7).astype("float32")
    df["amt_log"] = np.log1p(df.TransactionAmt)
    df["amt_cents"] = (df.TransactionAmt - np.floor(df.TransactionAmt)).astype("float32")

    # 还原用户：card1 + addr1 + 首次交易日（day - D1）
    d1n = np.floor(day - df.D1)
    df["uid"] = df.card1.astype(str) + "_" + df.addr1.astype(str) + "_" + d1n.astype(str)
    df["device"] = np.where(df.DeviceInfo.notna(),
                            df.DeviceInfo.astype(str) + "_" + df.id_31.astype(str) + "_" + df.id_33.astype(str),
                            "UNK_" + df.TransactionID.astype(str))  # 没有设备信息的不互相关联

    g = df.groupby("uid")
    df["uid_cnt"] = g.cumcount() + 1
    cs = g.TransactionAmt.cumsum()
    cs2 = (df.TransactionAmt ** 2).groupby(df.uid).cumsum()
    df["uid_amt_mean"] = cs / df.uid_cnt
    df["uid_amt_std"] = np.sqrt(np.maximum(cs2 / df.uid_cnt - df.uid_amt_mean ** 2, 0))
    df["amt_div_uid_mean"] = df.TransactionAmt / df.uid_amt_mean
    df["uid_gap"] = df.TransactionDT - g.TransactionDT.shift(1)
    df["uid_n_email"] = cum_distinct(df, "uid", "P_emaildomain")
    df["uid_n_product"] = cum_distinct(df, "uid", "ProductCD")
    for c in ("card1", "addr1", "P_emaildomain"):
        df[f"{c}_cnt"] = df.groupby(c, dropna=False).cumcount() + 1
    # 用户历史均值（累计，只用过去）：Vesta 的计数类 C 特征、距离
    for c in ("C1", "C13", "C14", "dist1"):
        df[f"uid_{c}_mean"] = (df[c].fillna(0).groupby(df.uid).cumsum() / df.uid_cnt).astype("float32")
    # D 列是“距某事件的天数”，随时间漂移；换成“事件发生的日期”后，同一用户的值稳定
    dnorm = []
    for c in ("D2", "D3", "D4", "D10", "D11", "D15"):
        df[f"{c}n"] = (np.floor(day) - df[c]).astype("float32")
        dnorm.append(f"{c}n")
    # 交易速度（velocity）：过去 1 小时 / 24 小时 / 7 天的笔数与金额
    t, amt = df.TransactionDT.values, df.TransactionAmt.values.astype(float)
    for name, w in (("1h", 3600), ("24h", 86400), ("7d", 7 * 86400)):
        df[f"uid_cnt_{name}"], df[f"uid_amt_{name}"] = window_stats(df.uid.values, t, amt, w)
    df["card1_cnt_24h"], _ = window_stats(df.card1.astype(str).values, t, amt, 86400)
    df["amt_div_uid_amt_7d"] = (df.TransactionAmt / (df.uid_amt_7d / df.uid_cnt_7d)).astype("float32")
    velocity = ["uid_cnt_1h", "uid_amt_1h", "uid_cnt_24h", "uid_amt_24h", "uid_cnt_7d", "uid_amt_7d",
                "card1_cnt_24h", "amt_div_uid_amt_7d"]
    behaviour = (["uid_cnt", "uid_amt_mean", "uid_amt_std", "amt_div_uid_mean", "uid_gap",
                  "uid_n_email", "uid_n_product", "card1_cnt", "addr1_cnt", "P_emaildomain_cnt",
                  "uid_C1_mean", "uid_C13_mean", "uid_C14_mean", "uid_dist1_mean"] + dnorm + velocity)

    # 设备关联（团伙信号）
    df["device_n_uid"] = cum_distinct(df, "device", "uid")
    df["uid_n_device"] = cum_distinct(df, "uid", "device")
    df["device_cnt"] = df.groupby("device").cumcount() + 1
    df["device_cnt_24h"], _ = window_stats(df.device.values, t, amt, 86400)
    df["device_n_card1"] = cum_distinct(df, "device", "card1")
    known = df.DeviceInfo.notna()
    for c in ("device_n_uid", "uid_n_device", "device_cnt", "device_cnt_24h", "device_n_card1"):
        df[c] = df[c].where(known, np.nan)
    graph = ["device_n_uid", "uid_n_device", "device_cnt", "device_cnt_24h", "device_n_card1"]

    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
                "DeviceInfo", "id_30", "id_31", "id_33"] + M_COLS
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    raw = [c for c in df.columns if c not in
           {"TransactionID", "isFraud", "TransactionDT", "uid", "device"} | set(behaviour) | set(graph)]
    return df, raw, behaviour, graph, [c for c in cat_cols if c in df.columns]


PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127, min_child_samples=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_cat_to_onehot=4, cat_smooth=50, seed=SEED, verbose=-1, n_jobs=-1)


def run(df, feats, cats, idx_tr, idx_va, idx_te, name):
    t0 = time.time()
    X, y = df[feats], df.isFraud
    dtr = lgb.Dataset(X.iloc[idx_tr], y.iloc[idx_tr], categorical_feature=[c for c in cats if c in feats])
    dva = lgb.Dataset(X.iloc[idx_va], y.iloc[idx_va], reference=dtr)
    m = lgb.train(PARAMS, dtr, 5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
    p_va = m.predict(X.iloc[idx_va], num_iteration=m.best_iteration)
    p_te = m.predict(X.iloc[idx_te], num_iteration=m.best_iteration)
    rep = {"n_features": len(feats), "best_iter": m.best_iteration,
           "valid": binary_report(y.iloc[idx_va], p_va),
           "test": binary_report(y.iloc[idx_te], p_te),
           "test_recall_at_alert": recall_at_alert_rate(y.iloc[idx_te].values, p_te),
           "seconds": round(time.time() - t0, 1)}
    print(f"[{name}] 特征 {len(feats)}  test AUC={rep['test']['auc']:.4f} KS={rep['test']['ks']:.4f} "
          f"PR-AUC={rep['test']['pr_auc']:.4f}  top5%召回={rep['test_recall_at_alert']['top5%']['recall']:.3f}")
    return m, rep, p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_v", action="store_true")
    a = ap.parse_args()
    set_seed()
    t0 = time.time()
    df = load(a.use_v)
    print(f"交易 {len(df)} 条，欺诈率 {df.isFraud.mean():.4f}")
    df, raw, beh, graph, cats = build_features(df)

    n = len(df)
    cut_te, cut_va = int(n * 0.8), int(n * 0.8 * 0.85)
    idx_tr, idx_va, idx_te = np.arange(cut_va), np.arange(cut_va, cut_te), np.arange(cut_te, n)
    days = df.TransactionDT / 86400
    split = {k: [round(float(days.iloc[i[0]]), 1), round(float(days.iloc[i[-1]]), 1), int(len(i))]
             for k, i in (("train", idx_tr), ("valid", idx_va), ("test", idx_te))}
    print("时间切分（天）:", split)

    res = {"split_days_[start,end,n]": split, "fraud_rate": float(df.isFraud.mean()), "ablation": {}}
    _, res["ablation"]["A_原始特征"], _, _ = run(df, raw, cats, idx_tr, idx_va, idx_te, "A 原始特征")
    _, res["ablation"]["B_+用户行为"], _, _ = run(df, raw + beh, cats, idx_tr, idx_va, idx_te, "B +用户行为")
    m, res["ablation"]["C_+设备关联"], p_va, p_te = run(df, raw + beh + graph, cats, idx_tr, idx_va, idx_te,
                                                     "C +设备关联")
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    res["top_features_gain"] = {k: round(float(v), 1) for k, v in imp.head(25).items()}
    res["new_feature_rank"] = {c: int(list(imp.index).index(c) + 1) for c in beh + graph}
    res["total_minutes"] = round((time.time() - t0) / 60, 1)
    save_json(res, RESULTS / "s04_fraud.json")
    np.savez(RESULTS / "probs_fraud.npz", dev=p_va, dev_y=df.isFraud.values[idx_va],
             test=p_te, test_y=df.isFraud.values[idx_te])
    print("重要性前 15:", list(imp.head(15).index))
    print("完成，结果: results/s04_fraud.json")


if __name__ == "__main__":
    main()
