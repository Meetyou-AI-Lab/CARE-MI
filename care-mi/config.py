import os
import sys

OPENAI_API_KEY = "" # Place your OpenAI key here.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
DATA = os.path.join(ROOT, "data")
CODE = os.path.join(ROOT, "care-mi")
CORPUS = os.path.join(ROOT, "corpus")

TEXTBOOK  = os.path.join(CORPUS, "textbook")
WIKIPEDIA = os.path.join(CORPUS, "wikipedia")
STOPWORDS = os.path.join(CORPUS, "stopwords")

BENCHMARK = os.path.join(DATA, "benchmark")

BIOS     = os.path.join(DATA, "BIOS")
CPUBMED  = os.path.join(DATA, "CPUBMED")
MEDQA    = os.path.join(DATA, "MEDQA")
MLECQA   = os.path.join(DATA, "MLECQA")
WORDLIST = os.path.join(DATA, "WORDLIST")

BIOS_SRC     = os.path.join(BIOS, "original")
CPUBMED_SRC  = os.path.join(CPUBMED, "original")
MEDQA_SRC    = os.path.join(MEDQA, "original")
MLECQA_SRC   = os.path.join(MLECQA, "original")

BIOS_EXT     = os.path.join(BIOS, "extracted")
CPUBMED_EXT  = os.path.join(CPUBMED, "extracted")
MEDQA_EXT    = os.path.join(MEDQA, "extracted")
MLECQA_EXT   = os.path.join(MLECQA, "extracted")

MLECQA_CATEGORY      = ["Clinic", "CWM", "PublicHealth", "Stomatology", "TCM"]
MLECQA_FILTERED_KEYS = ["qtext", "options", "answer"]
MLECQA_EXCLUDE_FROM_QUESTIONS = [
    "上述", "下述", "下列", "下面", "以下", "哪项", "一项", "一种", "哪种", "哪些",
    "错误", "有误", "正确", "不对的", "不宜", "不恰当", "不合适", "不应", "不会", "不能",
    "包括", "不属于", "除了", "无关", "不可能", "不是", "没有", "不存在"]
MLECQA_EXCLUDE_FROM_ANSWERS = ["以上全","以上各","以上都","以上均","包括以上","以上皆", "上述", "都不", "任何", "任意"]
MLECQA_AMBIGUOUS_PAIRS = {
    "keep": ["最常见", "最可能", "最主要", "最容易", "最恰当", "最合适", "最合理"],
    "drop": ["常",    "可能" ,  "主要",   "容易",   "恰当",   "合适",   "合理"],
}

MEDQA_KEYS          = ['question', 'options', 'answer', 'meta_info', 'answer_idx']
MEDQA_FILTERED_KEYS = ['question', 'options', 'answer_idx']