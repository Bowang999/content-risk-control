# -*- coding: utf-8 -*-
"""第 2 步：微调中文预训练模型做违规内容识别。

用法：
  python s02_text_bert.py                 # 原始训练
  python s02_text_bert.py --robust        # 归一化 + 对抗增强训练
有 GPU 默认用 hfl/chinese-roberta-wwm-ext（12 层）；只有 CPU 默认用 hfl/rbt3（3 层，快很多）。
国内网络建议先设置：set HF_ENDPOINT=https://hf-mirror.com （run_all.bat 已经设好）
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import best_f1_threshold, binary_report, load_cold, save_json
from config import RESULTS, set_seed
from perturb import build_homophone_table, normalize
from text_utils import ADV_TEST_PATH, adv_eval, augment_train


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--robust", action="store_true", help="归一化 + 对抗增强训练")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_train", type=int, default=0, help=">0 时只用这么多条训练样本（CPU 太慢时用）")
    return ap.parse_args()


@torch.no_grad()
def predict(model, tok, texts, device, max_len, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(list(texts[i:i + batch]), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    a = get_args()
    set_seed()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model_name = a.model or ("hfl/chinese-roberta-wwm-ext" if cuda else "hfl/rbt3")
    epochs = a.epochs or (3 if cuda else 2)
    batch = a.batch or (32 if cuda else 16)
    lr = a.lr or (2e-5 if "wwm" in model_name else 5e-5)
    tag = "robust" if a.robust else "raw"
    print(f"设备={device}  模型={model_name}  epochs={epochs} batch={batch} lr={lr} 模式={tag}")

    d = load_cold()
    tr, dv, te = d["train"], d["dev"], d["test"]
    adv = pd.read_csv(ADV_TEST_PATH, dtype={"text_orig": str, "text_adv": str}).fillna("")  # 由第 1 步生成
    prep = normalize if a.robust else (lambda s: s)

    if a.robust:
        cw = json.load(open(RESULTS / "char_weight.json", encoding="utf-8"))
        homo = build_homophone_table(tr.text.tolist())
        tr, n_extra = augment_train(tr, cw, homo)
        print(f"对抗增强后训练集 {len(tr)} 条（新增 {n_extra}）")
    if a.max_train and len(tr) > a.max_train:
        tr = tr.sample(n=a.max_train, random_state=0).reset_index(drop=True)
        print(f"只用 {len(tr)} 条训练样本")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    train_texts = [prep(t) for t in tr.text]
    train_labels = tr.label.tolist()

    def collate(batch_idx):
        enc = tok([train_texts[i] for i in batch_idx], truncation=True, max_length=a.max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([train_labels[i] for i in batch_idx])
        return enc

    loader = DataLoader(list(range(len(train_texts))), batch_size=batch, shuffle=True,
                        collate_fn=collate, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    except (AttributeError, TypeError):  # 旧版 torch
        scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    dev_texts = [prep(t) for t in dv.text]
    best_auc, best_state, history = -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for step, b in enumerate(loader):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type="cuda", enabled=cuda):
                loss = model(**b).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  epoch {ep+1} step {step}/{len(loader)} loss={loss.item():.4f} 已用 {el/60:.1f} 分钟")
        p_dev = predict(model, tok, dev_texts, device, a.max_len)
        rep = binary_report(dv.label, p_dev)
        history.append({"epoch": ep + 1, **{k: rep[k] for k in ("auc", "f1", "macro_f1")}})
        print(f"epoch {ep+1} dev AUC={rep['auc']:.4f} F1={rep['f1']:.4f}")
        if rep["auc"] > best_auc:
            best_auc, best_state = rep["auc"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    p_dev = predict(model, tok, dev_texts, device, a.max_len)
    th = best_f1_threshold(dv.label, p_dev)
    p_test = predict(model, tok, [prep(t) for t in te.text], device, a.max_len)
    p_ao = predict(model, tok, [prep(t) for t in adv.text_orig], device, a.max_len)
    p_aa = predict(model, tok, [prep(t) for t in adv.text_adv], device, a.max_len)

    # 按话题（种族/地域/性别）拆分测试集表现
    by_topic = {}
    if "topic" in te.columns:
        for tp, g in te.assign(p=p_test).groupby("topic"):
            by_topic[tp] = binary_report(g.label, g.p, th)

    res = {"model": model_name, "device": str(device), "mode": tag, "epochs": epochs,
           "batch": batch, "lr": lr, "n_train": len(tr), "history": history,
           "dev": binary_report(dv.label, p_dev, th), "test": binary_report(te.label, p_test, th),
           "test_by_topic": by_topic, "adversarial": adv_eval(adv, p_ao, p_aa, th),
           "train_minutes": round((time.time() - t0) / 60, 1)}
    save_json(res, RESULTS / f"s02_bert_{tag}.json")
    np.savez(RESULTS / f"probs_bert_{tag}.npz", dev=p_dev, dev_y=dv.label.values,
             test=p_test, test_y=te.label.values, threshold=th)
    print("test:", {k: round(v, 4) for k, v in res["test"].items() if isinstance(v, float)})
    for m, v in res["adversarial"].items():
        if v["recall_orig"] is not None:
            print(f"  {m:14s} 召回 {v['recall_orig']:.3f} -> {v['recall_adv']:.3f} | 绕过 {v['evasion_rate']:.3f}")
    print(f"完成，结果: results/s02_bert_{tag}.json")


if __name__ == "__main__":
    main()
