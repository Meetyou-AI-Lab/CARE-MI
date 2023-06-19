import stanza
import utils
import argparse
import config as cfg
import pandas as pd
import numpy as np
from os.path import join, exists
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

def load_corpus(corpus="wikipedia", length_threshold=10):
    assert corpus in ["wikipedia", "textbook"]
    if corpus == "wikipedia":
        folder = cfg.WIKIPEDIA
    elif corpus == "textbook":
        folder = cfg.TEXTBOOK
    fp = join(folder, "paragraphs.txt")
    with open(fp, "r", encoding="utf-8") as reader:
        book = reader.readlines()
    book = [i.strip() for i in book]
    if length_threshold is not None:
        book = [i for i in book if len(i) > length_threshold]
    print(f"{corpus.title()} paragraphs loaded. {len(book)} paragraphs in total.")
    return book

def tokenize_(processor: stanza.Pipeline, document: str) -> list:
    doc = processor(document)
    sents = doc.sentences
    words = [[w.text for w in s.words] for s in sents]
    words = [item for sublist in words for item in sublist]
    return words


def preprocess_query_(processor: stanza.Pipeline, query: str, stopwords: set) -> list:
    query_tokens = tokenize_(processor, query)
    query_tokens = [t for t in query_tokens if t not in stopwords]
    return query_tokens

def pretokenize_corpus_(processor: stanza.Pipeline, stopwords: set, corpus="textbook"):
    assert corpus in ["wikipedia", "textbook"]
    if corpus == "wikipedia":
        folder = cfg.WIKIPEDIA
    elif corpus == "textbook":
        folder = cfg.TEXTBOOK
    corpus = load_corpus(corpus)
    cur_fp = join(folder, "tokenized_paragraphs.tsv")
    if exists(cur_fp):
        print("Tokenized data already created.")
        data = utils.load_sheet(cur_fp, converters={"tokens": utils.convert_list_str_to_list})
        assert len(corpus) == len(data)
    else:
        print("Doing corpus tokenization now...")
        tokenize_corpus = [tokenize_(processor, sent) for sent in tqdm(corpus)]                   # tokenized
        tokenize_corpus = [[w for w in words if w not in stopwords] for words in tokenize_corpus] # stopwords
        data = pd.DataFrame()
        data["tokens"] = tokenize_corpus
        utils.save_sheet(data, cur_fp)
    return data

def retrieval_bm25(queries: list, corpus='textbook', retriever="BM25Okapi", n=3):
    if   retriever == "BM25Okapi": retriever = BM25Okapi
    elif retriever == "BM25L":     retriever = BM25L
    elif retriever == "BM25Plus":  retriever = BM25Plus
    else: raise NotImplementedError(f"Retrieval functions not found in `rank_bm25`.")
    processor = stanza.Pipeline("zh-hans", processors="tokenize", tokenize_with_jieba=True)
    stopwords = set(utils.load_stopwords())
    query_tokens = [preprocess_query_(processor, q, stopwords) for q in queries]
    data      = pretokenize_corpus_(processor, stopwords, corpus)
    corpus    = load_corpus(corpus)
    assert len(data)   == len(corpus)
    tokenized_corpus   = data["tokens"].tolist()
    assert len(corpus) == len(tokenized_corpus), "The length of corpus and tokenized corpus have to be the same."
    trained_retriever  = retriever(tokenized_corpus)
    print(f"Retrieving documents:")
    scores = [trained_retriever.get_scores(q) for q in tqdm(query_tokens)]
    scores = np.array(scores)
    top_n_idxs = [np.argsort(ss)[::-1][:n] for ss in scores]
    top_n_docs = [[corpus[i] for i in idxs] for idxs in top_n_idxs]
    return top_n_docs

def main(args):
    fp   = join(cfg.BENCHMARK, "questions_original.tsv")
    data = utils.load_sheet(fp)
    queries = []
    for idx, col in enumerate(['context', 'answer']):
        if idx == 0:
            queries = data[col].tolist()
        else:
            cur_ = data[col].tolist()
            queries = [f"{queries[i]}{cur_[i]}" for i in range(len(cur_))]
            queries = [q.replace('nan', '') for q in queries]
    top_n_docs = retrieval_bm25(
        queries=queries,
        corpus=args.corpus,
        retriever=args.retriever,
        n=args.n)
    for i in range(args.n):
        data[f"{args.corpus}_{i + 1}"] = utils.padding_column([docs[i] for docs in top_n_docs], max_len=len(data))
    save_fp = join(cfg.BENCHMARK, "questions_retrieval.tsv")
    utils.save_sheet(data, save_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Knowledge retrieval.")
    parser.add_argument('--corpus',    type=str, default='wikipedia', choices=['wikipedia', 'textbook'])
    parser.add_argument('--retriever', type=str, default='BM25Okapi', choices=["BM25Okapi", "BM25L", "BM25Plus"])
    parser.add_argument('--n',         type=int, default=3,           help="Top n")
    args = parser.parse_args()
    main(args)