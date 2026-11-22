# -*- coding: utf-8 -*-
"""项目全局配置：路径、随机种子、决策层目标。"""
from pathlib import Path
import os
import random

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COLD_DIR = DATA / "COLD"            # 放 COLDataset 的 train.csv / dev.csv / test.csv
IEEE_DIR = DATA / "ieee-fraud"      # 放 Kaggle IEEE-CIS 的 train_transaction.csv / train_identity.csv
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
for d in (RESULTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# 决策层目标（在验证集上选阈值，在测试集上报告）
DECISION_TARGETS = {
    # 内容：自动处置要求精确率 >= 95%（误杀少），整体召回（自动处置 + 人审）>= 95%
    "content": {"auto_precision": 0.95, "pass_recall": 0.95},
    # 交易/账号：欺诈样本稀少，自动拦截精确率要求放到 80%，整体召回 >= 80%
    "fraud": {"auto_precision": 0.80, "pass_recall": 0.80},
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
