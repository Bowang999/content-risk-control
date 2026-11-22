# -*- coding: utf-8 -*-
"""第 3 步（可选）：本地大模型零样本审核，和微调模型对比。
需要先装 Ollama 并拉一个中文模型，例如：  ollama pull qwen2.5:7b
用法： python s03_llm_zero_shot.py --model qwen2.5:7b --n 300 --n_adv 150
"""
import argparse
import json
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from common import binary_report, load_cold, save_json
from config import RESULTS, SEED
from text_utils import ADV_TEST_PATH

PROMPT = """你是中文社区的内容安全审核员。判断下面这条评论是否属于冒犯性内容。
冒犯性内容包括：辱骂、人身攻击、对某个群体（地域、种族、性别等）的歧视或贬低、带有恶意的刻板印象。
注意：评论可能用谐音字、拼音、拆字、插入符号等方式规避审核，请还原其真实含义后再判断。
只回答一个字：“是”（冒犯）或“否”（不冒犯）。

评论：{text}
回答："""

URL = "http://localhost:11434/api/generate"


def ask(model, text):
    body = json.dumps({"model": model, "prompt": PROMPT.format(text=text), "stream": False,
                       "options": {"temperature": 0, "num_predict": 5}}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        ans = json.loads(r.read().decode("utf-8"))["response"]
    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
    if ans.startswith("是"):
        return 1
    if ans.startswith("否"):
        return 0
    return 1 if "是" in ans[:3] else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_adv", type=int, default=150)
    a = ap.parse_args()
    try:
        tags = json.loads(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5).read())
        names = [m["name"] for m in tags.get("models", [])]
        print("本地模型:", names)
        if a.model not in names:
            print(f"没找到 {a.model}，请先 ollama pull {a.model}，或用 --model 指定已有模型。跳过此步。")
            return
    except Exception as e:
        print("连不上 Ollama（http://localhost:11434），跳过此步。", e)
        return

    te = load_cold()["test"]
    samp = pd.concat([te[te.label == v].sample(n=min((te.label == v).sum(), a.n // 2), random_state=SEED)
                      for v in (0, 1)])
    adv = pd.read_csv(ADV_TEST_PATH, dtype={"text_orig": str, "text_adv": str}).fillna("")
    adv = adv[adv.label == 1].sample(n=min(a.n_adv, (adv.label == 1).sum()), random_state=SEED)

    t0 = time.time()
    pred = []
    for i, t in enumerate(samp.text):
        pred.append(ask(a.model, t))
        if i % 50 == 0:
            print(f"  干净样本 {i}/{len(samp)}")
    adv_orig = [ask(a.model, t) for t in adv.text_orig]
    adv_pred = [ask(a.model, t) for t in adv.text_adv]
    per_call = (time.time() - t0) / (len(samp) + 2 * len(adv))

    adv = adv.assign(o=adv_orig, p=adv_pred)
    adv_res = {m: {"n": int(len(g)), "recall_orig": float(g.o.mean()), "recall_adv": float(g.p.mean())}
               for m, g in list(adv.groupby("method")) + [("ALL", adv)]}
    res = {"model": a.model, "clean_sample": binary_report(samp.label, np.array(pred, float), 0.5),
           "adversarial_offensive": adv_res, "seconds_per_call": round(per_call, 2),
           "note": "大模型只输出是/否，没有连续分数，所以没有 AUC；样本为测试集分层抽样",
           "sample_idx": [int(i) for i in samp.index], "adv_rows": [int(i) for i in adv.index]}
    save_json(res, RESULTS / "s03_llm.json")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
