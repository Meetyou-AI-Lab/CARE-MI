import json
import ast
import os
import config as cfg
import pandas as pd

def save_sheet(data: pd.DataFrame, fp: str):
    dir_path = os.path.dirname(fp)
    postfix  = fp.split('.')[-1]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if postfix   == 'csv': data.to_csv(fp, index=False)
    elif postfix == 'xlsx': data.to_excel(fp, index=False)
    elif postfix == 'tsv': data.to_csv(fp, sep='\t', index=False)
    else: raise NotImplementedError

def load_sheet(fp, converters=None, encoding='utf-8'):
    postfix = fp.split('.')[-1]
    if postfix == 'xlsx':  data = pd.read_excel(fp, converters=converters)
    elif postfix == 'csv': data = pd.read_csv(fp, sep=',', header=0, converters=converters, encoding=encoding)
    elif postfix == 'tsv': data = pd.read_csv(fp, sep='\t', header=0, converters=converters, encoding=encoding)
    else: raise NotImplementedError
    return data

def save_jsonl(list_obj, dst):
    with open(dst, 'w', encoding='utf-8') as f:
        for d in list_obj:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

def load_json(f_path) -> dict:
    with open(f_path, 'r', encoding='utf-8') as reader:
        dic = json.load(reader)
    return dic

def load_jsonl(file_path, encoding='utf-8'):
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def filter_dict_keys_(dic: dict, keys=cfg.MLECQA_FILTERED_KEYS) -> dict:
    return {k: dic[k] for k in keys if k in dic}

def convert_list_str_to_list(list_str):
    try:
        return ast.literal_eval(list_str)
    except (SyntaxError, ValueError) as e:
        print(e)
        return list_str
    
def remove_question_mark_(text: list) -> list:
    text = [sent.replace('?', '') for sent in text]
    text = [sent.replace('？', '') for sent in text]
    return text

def levenshtein_distance(str1, str2):
    # Calculates the Levenshtein distance between two strings.
    m, n = len(str1), len(str2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
    for j in range(1, n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,                     # Deletion
                d[i][j - 1] + 1,                     # Insertion
                d[i - 1][j - 1] + substitution_cost) # Substitution
    return d[m][n]

def load_stopwords():
    folder    = cfg.STOPWORDS
    all_words = []
    for fn in os.listdir(folder):
        if fn.split(".")[-1] != 'txt':
            continue
        fp = os.path.join(folder, fn)
        with open(fp, 'r', encoding='utf-8') as reader:
            words = reader.readlines()
            words = [i.strip() for i in words]
            all_words += words
    all_words = list(set(all_words))
    print(f"The number of unique stopwords is {len(all_words)}.")
    return all_words

def padding_column(lis: list, max_len: int, pad_token='') -> list:
    if len(lis) >= max_len:
        return lis[: max_len]
    lis = lis + [pad_token] * (max_len - len(lis))
    return lis