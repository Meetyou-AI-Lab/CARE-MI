import re
import random
import argparse
import preprocess
import utils
import config as cfg
import numpy as np
from config import *
from os.path import join

def replace_last_appear_(statement:str, pattern:str, replace_item:str):
    """
    Only replace the last appearing `pattern` from the `statement`
    and replace it with the `replace_item`.
    """
    if "+" in statement:
        statement = statement.split("+")
        statement = "加".join(statement)
    if "+" in pattern:
        pattern = pattern.split("+")
        pattern = "加".join(pattern)
    last_match = None
    for match in re.finditer(pattern, statement):
        last_match = match
    if last_match:
        start, end = last_match.start(), last_match.end()
        return statement[:start] + replace_item + statement[end:]
    else:
        raise ValueError

def extract_similar_substring_(source: str, string: str):
    # Retrieve a most similar substring from the source that matches the given string.
    start_index, end_index = [], []
    for i in range(0, len(source)):
        if source[i] == string[0]:
            start_index.append(i)
        if source[i] == string[-1]:
            end_index.append(i)
    if len(start_index) >= 1 and len(end_index) >= 1:
        start, end = start_index[-1], end_index[-1]
    else: return np.nan
    range_len = end + 1 - start
    if start >= end or (range_len < len(string) - 2) or (range_len > len(string) + 2):
        return np.nan
    return source[start: end + 1]

def replacement_(statement: str, answer: str, wrong_answers: list, wrong_answer_sample_method='first') -> str:
    assert wrong_answer_sample_method in ['first', 'random']
    if wrong_answer_sample_method == 'first':
        wrong_answer = wrong_answers[0]
    else:
        wrong_answer = random.choice(wrong_answers)
    false_statement = np.nan
    try:
        false_statement = replace_last_appear_(statement, answer, wrong_answer)
    except(ValueError):
        try:
            false_statement = re.sub(answer, wrong_answer, statement)
        except:
            try:
                false_statement = statement.replace(answer, wrong_answer)
            except:
                false_statement = np.nan
    if false_statement == statement or false_statement == np.nan:
        substring = extract_similar_substring_(statement, answer)
        if isinstance(substring, str) and utils.levenshtein_distance(substring, answer) <= 2:
            false_statement = statement.replace(substring, wrong_answer)
            print(f"Replacement Succeed: {statement}--{answer}--{wrong_answer}")
        else:
            print(f"Replacement Failed:  {statement}--{answer}--{wrong_answer}")
        false_statement = np.nan
    return false_statement

def batch_replacement_(statements, answers, wrong_answers):
    false_statements = []
    for i in range(len(statements)):
        false_statement = replacement_(statements[i], answers[i], wrong_answers[i])
        false_statements.append(false_statement)
    return false_statements

def bios_cid2term_():
    concept_terms = preprocess.load_bios_conceptterms()
    cids  = concept_terms['CID']
    terms = concept_terms['STR']
    cid2term = {}
    for cid, term in zip(cids, terms):
        if cid in cid2term:
            cid2term[cid].append(term)
        else:
            cid2term[cid] = [term]
    return cid2term

def bios_select_wrong_answer_(cid, cid2term):
    assert cid in cid2term
    cids = list(cid2term.keys())
    wrong_cid = random.choice(cids)
    try: wrong_terms = cid2term[wrong_cid]
    except:
        while wrong_cid == cid or wrong_cid not in cid2term:
            wrong_cid = random.choice(cids)
            wrong_terms = cid2term[wrong_cid]
    wrong_term = random.choice(wrong_terms)
    return str(wrong_term)

def cpubmed_select_wrong_answer_(entity, all_entities):
    wrong_entity = random.choice(all_entities)
    while wrong_entity == entity:
        wrong_entity = random.choice(all_entities)
    return str(wrong_entity)

def main(args):
    if args.dataset == "BIOS":
        folder  = cfg.BIOS
        fp = join(folder, "statements.tsv")
        cid2term = bios_cid2term_()
        data     = utils.load_sheet(fp)
        answers  = data['TERM_TAIL'].tolist()
        cids     = data['CID_TAIL'].tolist()
        print("Retrieving wrong answers for BIOS statements:")
        wrong_answers = [[bios_select_wrong_answer_(a, cid2term)] for a in cids]
    elif args.dataset == "CPUBMED":
        folder  = cfg.CPUBMED
        fp = join(folder, "statements.tsv")
        data    = utils.load_sheet(fp)
        rels    = data["REL"].tolist()
        heads   = data["HEAD_ENT"].tolist()
        answers = data["TAIL_ENT"].tolist()
        for i in range(len(rels)):
            if rels[i] in {'预防', '传播途径'}:
                answers[i] = heads[i]
        all_entites = preprocess.load_cpubmed()['TAIL_ENT'].tolist()
        print("Retrieving wrong answers for CPubMed statements:")
        wrong_answers = [[cpubmed_select_wrong_answer_(a, all_entites)] for a in answers]
    else:
        if args.dataset == "MEDQA":
            folder = cfg.MEDQA
        else:
            folder = cfg.MLECQA
        fp   = join(folder, "statements.tsv")
        data = utils.load_sheet(fp, converters={'wrong_answer': utils.convert_list_str_to_list})
        wrong_answers = data['wrong_answer'].tolist()
        answers = data['answer'].tolist()
    statements = data["statement"].tolist()
    false_statements = batch_replacement_(statements, answers, wrong_answers)
    data[f'statement-rep'] = false_statements
    save_fp = join(folder, "statements-rep.tsv")
    utils.save_sheet(data, save_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate false declarative sentences using true ones.')
    parser.add_argument('--dataset', type=str, default="MEDQA", choices=["BIOS", "CPUBMED", "MEDQA", "MLECQA"])
    args = parser.parse_args()
    main(args)