# -*- coding: utf-8 -*-
"""黑产常见的文本对抗变体：插符号、同音字替换、拼音替换、拆字；以及推理前的文本归一化。

扰动位置选择：优先选“敏感字”（由基线模型学到的单字权重决定），模拟黑产专门改写敏感词的行为；
没有权重时退化为随机位置。
"""
import random
import re
import unicodedata

try:
    from pypinyin import lazy_pinyin
    HAS_PINYIN = True
except ImportError:  # 没装 pypinyin 时，同音字/拼音两种变体会被跳过
    HAS_PINYIN = False

CJK = re.compile(r"[一-鿿]")
NOISE_SYMBOLS = list("*#@&%$~^.-_|/\\!·、 ")

# 常见可拆字（左右/上下结构），用于“拆字”变体
SPLIT_MAP = {
    "妈": "女马", "好": "女子", "她": "女也", "他": "亻也", "你": "亻尔", "们": "亻门",
    "林": "木木", "明": "日月", "炮": "火包", "狗": "犭句", "猪": "犭者", "骗": "马扁",
    "婊": "女表", "嫖": "女票", "鸡": "又鸟", "吗": "口马", "吧": "口巴", "屌": "尸吊",
    "死": "歹匕", "蠢": "春虫虫", "贱": "贝戋", "骚": "马蚤", "娘": "女良", "妹": "女未",
    "姐": "女且", "奴": "女又", "粪": "米共", "吃": "口乞", "讨": "讠寸", "厌": "厂犬",
    "坏": "土不", "打": "扌丁", "揍": "扌奏", "种": "禾中", "恨": "忄艮", "怪": "忄圣",
    "物": "牛勿", "狼": "犭良", "猴": "犭侯", "蛆": "虫且", "蝗": "虫皇", "贼": "贝戎",
    "滚": "氵衮", "渣": "氵查", "话": "讠舌", "说": "讠兑", "认": "讠人", "地": "土也",
    "域": "土或", "河": "氵可", "湾": "氵弯", "汉": "氵又", "姆": "女母", "妓": "女支",
    "婚": "女昏", "嫁": "女家", "猫": "犭苗", "粗": "米且", "瞎": "目害", "疯": "疒风",
    "病": "疒丙", "残": "歹戋", "杂": "九木", "奸": "女干", "妖": "女夭", "娼": "女昌",
}


def build_homophone_table(texts):
    """用语料里出现过的汉字，建 拼音 -> 同音字列表。"""
    if not HAS_PINYIN:
        return {}
    chars = set()
    for t in texts:
        chars.update(CJK.findall(t))
    table = {}
    for ch in chars:
        py = lazy_pinyin(ch)[0]
        table.setdefault(py, []).append(ch)
    return {k: sorted(v) for k, v in table.items() if len(v) > 1}


def pick_positions(text, char_weight=None, ratio=0.2, rng=random, allowed=None):
    """选择要扰动的汉字位置。char_weight: 单字 -> 违规权重；优先改权重最高的字。"""
    idx = [i for i, ch in enumerate(text) if CJK.match(ch) and (allowed is None or ch in allowed)]
    if not idx:
        return []
    k = max(1, int(round(len(idx) * ratio)))
    if char_weight:
        scored = [(char_weight.get(text[i], 0.0), rng.random(), i) for i in idx]
        scored.sort(reverse=True)
        chosen = [i for w, _, i in scored[:k]]
    else:
        chosen = rng.sample(idx, min(k, len(idx)))
    return sorted(chosen)


def insert_symbols(text, pos, rng=random):
    out = []
    pos = set(pos)
    for i, ch in enumerate(text):
        out.append(ch)
        if i in pos and i < len(text) - 1:
            out.append(rng.choice(NOISE_SYMBOLS))
    return "".join(out)


def homophone(text, pos, table, rng=random):
    if not HAS_PINYIN or not table:
        return text
    chars = list(text)
    for i in pos:
        py = lazy_pinyin(chars[i])[0]
        cands = [c for c in table.get(py, []) if c != chars[i]]
        if cands:
            chars[i] = rng.choice(cands)
    return "".join(chars)


def to_pinyin(text, pos, rng=random):
    if not HAS_PINYIN:
        return text
    chars = list(text)
    for i in pos:
        py = lazy_pinyin(chars[i])[0]
        chars[i] = py[0] if rng.random() < 0.5 else py  # 首字母（如 sb）或全拼
    return "".join(chars)


def split_chars(text, pos):
    chars = list(text)
    for i in pos:
        chars[i] = SPLIT_MAP.get(chars[i], chars[i])
    return "".join(chars)


METHODS = ["insert_symbol", "homophone", "pinyin", "split_char"]


def perturb(text, method, char_weight=None, homo_table=None, ratio=0.2, rng=random):
    if method == "split_char":
        pos = pick_positions(text, char_weight, ratio, rng, allowed=SPLIT_MAP)
        return split_chars(text, pos)
    pos = pick_positions(text, char_weight, ratio, rng)
    if method == "insert_symbol":
        return insert_symbols(text, pos, rng)
    if method == "homophone":
        return homophone(text, pos, homo_table, rng)
    if method == "pinyin":
        return to_pinyin(text, pos, rng)
    raise ValueError(method)


_KEEP = re.compile(r"[一-鿿A-Za-z0-9]")


def normalize(text):
    """推理前归一化：全角转半角、转小写、去掉夹在字里的符号/空白。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if _KEEP.match(ch))
