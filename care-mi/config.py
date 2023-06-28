from os.path import join, dirname, abspath
import sys

OPENAI_API_KEY = "" # Place your OpenAI key here.

ROOT =  dirname(dirname(abspath(__file__)))
sys.path.append(ROOT)
DATA = join(ROOT, "data")
CODE = join(ROOT, "care-mi")
CORPUS = join(ROOT, "corpus")

TEXTBOOK  = join(CORPUS, "textbook")
WIKIPEDIA = join(CORPUS, "wikipedia")
STOPWORDS = join(CORPUS, "stopwords")

BENCHMARK = join(DATA, "benchmark")

BIOS     = join(DATA, "BIOS")
CPUBMED  = join(DATA, "CPUBMED")
MEDQA    = join(DATA, "MEDQA")
MLECQA   = join(DATA, "MLECQA")
WORDLIST = join(DATA, "WORDLIST")

BIOS_SRC     = join(BIOS, "original")
CPUBMED_SRC  = join(CPUBMED, "original")
MEDQA_SRC    = join(MEDQA, "original")
MLECQA_SRC   = join(MLECQA, "original")

BIOS_EXT     = join(BIOS, "extracted")
CPUBMED_EXT  = join(CPUBMED, "extracted")
MEDQA_EXT    = join(MEDQA, "extracted")
MLECQA_EXT   = join(MLECQA, "extracted")

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