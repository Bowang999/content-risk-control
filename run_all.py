# -*- coding: utf-8 -*-
"""一键运行全部步骤，输出同时写进 results/run_log.txt。

  python run_all.py                 # 全部步骤
  python run_all.py --only 1 4 5    # 只跑指定步骤
  python run_all.py --max_train 8000   # CPU 太慢时，BERT 只用 8000 条训练
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOG = ROOT / "results" / "run_log.txt"

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", type=int, default=None)
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--llm", default="qwen2.5:7b")
a = ap.parse_args()

mt = ["--max_train", str(a.max_train)] if a.max_train else []
STEPS = [
    (1, "内容层基线 + 构造对抗测试集", ["s01_text_baseline.py"]),
    (2, "微调中文预训练模型（原始训练）", ["s02_text_bert.py"] + mt),
    (2, "微调中文预训练模型（归一化 + 对抗增强）", ["s02_text_bert.py", "--robust"] + mt),
    (3, "本地大模型零样本审核（可选，需要 Ollama）", ["s03_llm_zero_shot.py", "--model", a.llm]),
    (4, "交易/账号层反欺诈 LightGBM", ["s04_fraud_lgb.py"]),
    (5, "决策层三档分流", ["s05_decision.py"]),
]

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
LOG.parent.mkdir(exist_ok=True)
with open(LOG, "a", encoding="utf-8") as log:
    log.write(f"\n===== 运行开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    for num, name, cmd in STEPS:
        if a.only and num not in a.only:
            continue
        head = f"\n########## 第 {num} 步：{name} ##########"
        print(head)
        log.write(head + "\n")
        t0 = time.time()
        p = subprocess.Popen([sys.executable, "-u"] + cmd, cwd=SRC, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        p.wait()
        tail = f"---- 第 {num} 步结束，返回码 {p.returncode}，用时 {(time.time()-t0)/60:.1f} 分钟"
        print(tail)
        log.write(tail + "\n")
        if p.returncode != 0 and num in (1,):
            print("第 1 步失败，后续步骤依赖它，停止。请把 results/run_log.txt 发给 Claude。")
            break
print(f"\n全部结束。日志: {LOG}")
