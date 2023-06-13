import os
import time
import csv
import re
import utils
import config as cfg
import pandas as pd
import numpy as np

def load_topic_words(filter: bool=True) -> list:
    fn = 'combined.txt' if not filter else 'filtered.txt' 
    fp = os.path.join(cfg.WORDLIST, fn)
    with open(fp, 'r', encoding='utf-8') as reader:
        words = reader.readlines()
    words = [w.strip() for w in words]
    words = list(set(words))
    words = sorted(words)
    return words

def load_bios_conceptterms() -> pd.DataFrame:
    start_time = time.time()
    conceptterms_path = os.path.join(cfg.BIOS_SRC, "ConceptTerms_zh.csv")
    if not os.path.join(conceptterms_path):
        conceptterms_all = pd.read_csv(
            os.path.join(cfg.BIOS_SRC, "ConceptTerms.txt"),
            sep="|",
            usecols=[0,1,2,3],
            header=None,
            dtype={0: str, 1: str, 2: str, 3: "category"})
        conceptterms_all.columns = ['CID', 'SID', 'STR', 'LANG']
        conceptterms_zh = conceptterms_all[conceptterms_all.LANG == 'CHS'].drop(columns=['LANG'])
        conceptterms_zh.drop_duplicates(subset=['SID', 'STR'], keep=False)
        conceptterms_zh.to_csv(conceptterms_path, index=False)
    conceptterms = pd.read_csv(conceptterms_path, sep=",", engine='pyarrow')
    conceptterms = conceptterms[['CID', 'STR']]
    end_time = time.time()
    print(f"Loading BIOS conceptterms took {(end_time - start_time):.2f} seconds.")
    return conceptterms

def load_bios_relations() -> pd.DataFrame:
    start_time = time.time()
    relations = pd.read_csv(os.path.join(cfg.BIOS_SRC, "Relations.txt"),
        sep="|", usecols=[1,2,3], dtype={1: str, 2: str, 3: "category"},
        header=None)
    relations.columns = ['CID_HEAD', 'CID_TAIL', 'RELID']
    relations = relations.astype({"RELID": 'int32'})
    relations = relations.reset_index(drop=True)
    end_time = time.time()
    print(f"Loading BIOS relations took {(end_time - start_time):.2f} seconds.")
    return relations

def load_bios_relationnames() -> pd.DataFrame:
    relationname_path = os.path.join(cfg.BIOS_SRC, "RelationNames.txt")
    relationname = pd.read_csv(relationname_path, sep="|", header=None)
    relationname.columns = ['RELID', 'LANG', 'REL']
    relationname = relationname.astype({"RELID": 'int32'})
    relationname = relationname[relationname.LANG == "CHS"].drop(columns=['LANG'])
    relationname = relationname.reset_index(drop=True)
    return relationname

def load_bios_semantictypes() -> pd.DataFrame:
    semantictypes_path = os.path.join(cfg.BIOS_SRC, "SemanticTypes.txt")
    semantictypes = pd.read_csv(semantictypes_path, sep="|", usecols=[0, 1], header=None)
    semantictypes.columns = ['CID', 'STYID']
    semantictypes = semantictypes.astype({"STYID": 'int32'})
    semantictypes = semantictypes.reset_index(drop=True)
    return semantictypes

def load_bios_semanticnetwork() -> pd.DataFrame:
    semanticnetwork_path = os.path.join(cfg.BIOS_SRC, "SemanticNetwork.txt")
    semanticnetwork = pd.read_csv(semanticnetwork_path, sep='|', usecols=[0, 1, 2], header=None)
    semanticnetwork.columns = ['STYID', 'LANG', 'STY']
    semanticnetwork = semanticnetwork.astype({"STYID": 'int32'})
    semanticnetwork = semanticnetwork[semanticnetwork.LANG == "CHS"].drop(columns=['LANG'])
    semanticnetwork = semanticnetwork.reset_index(drop=True)
    return semanticnetwork

def load_filtered_bios_relations() -> pd.DataFrame:
    relations_fp = os.path.join(cfg.BIOS, f'relations.csv')
    if not os.path.exists(relations_fp):
        conceptterms = load_bios_conceptterms()
        seeds = set(load_topic_words())
        conceptterms['flag'] = conceptterms['STR'].apply(lambda text: text in seeds)
        conceptterms  = conceptterms[conceptterms['flag'] == True].drop(columns=['flag'])
        all_relations = load_bios_relations()
        concepts  = list(set(conceptterms['CID'].tolist()))
        relations = all_relations[(all_relations['CID_HEAD'].isin(concepts))&(all_relations['CID_TAIL'].isin(concepts))]
        utils.save_sheet(relations, relations_fp)
    else:
        relations = utils.load_sheet(relations_fp, header=0)
    return relations

def load_filtered_bios_metainfo() -> pd.DataFrame:
    metainfo_fp = os.path.join(cfg.BIOS, f'metainfo.tsv')
    if not os.path.exists(metainfo_fp):
        relations = load_filtered_bios_relations()
        conceptterms = load_bios_conceptterms()
        rel_cids = set(relations['CID_HEAD'].tolist() + relations['CID_TAIL'].tolist())
        all_cids = set(conceptterms['CID'].tolist()) & rel_cids
        relations = relations.loc[relations['CID_HEAD'].isin(all_cids)]
        relations = relations.loc[relations['CID_TAIL'].isin(all_cids)]
        cid_heads = relations['CID_HEAD'].tolist()
        cid_tails = relations['CID_TAIL'].tolist()
        relation_ids   = relations['RELID'].tolist()
        relation_dic   = load_bios_relationnames()
        relation_dic   = {relid: rel for relid, rel in zip(relation_dic['RELID'].tolist(), relation_dic['REL'].tolist())}
        relation_names = [relation_dic[relid] for relid in relation_ids]
        relations['REL'] = relation_names

        conceptterms = conceptterms.loc[conceptterms['CID'].isin(all_cids)]
        conceptterms = conceptterms.drop_duplicates(subset='CID', ignore_index=True)
        conceptterms = conceptterms[['CID', 'STR']]
        conceptterms = {cid: str for cid, str in zip(conceptterms['CID'].tolist(),conceptterms['STR'].tolist())}
        relations['TERM_HEAD'] = [conceptterms[cid] for cid in cid_heads]
        relations['TERM_TAIL'] = [conceptterms[cid] for cid in cid_tails]

        semantics = load_bios_semantictypes()
        semantics = semantics.loc[semantics['CID'].isin(all_cids)]
        semantics = {cid: styid for cid, styid in zip(semantics['CID'].tolist(), semantics['STYID'].tolist())}
        relations['STYID_HEAD'] = [semantics[cid] for cid in cid_heads]
        relations['STYID_TAIL'] = [semantics[cid] for cid in cid_tails]
        semantic_dic = load_bios_semanticnetwork()
        semantic_dic = {styid: rel for styid, rel in zip(semantic_dic['STYID'].tolist(), semantic_dic['STY'].tolist())}
        relations['STY_HEAD'] = [semantic_dic[styid] for styid in relations['STYID_HEAD'].tolist()]
        relations['STY_TAIL'] = [semantic_dic[styid] for styid in relations['STYID_TAIL'].tolist()]
        
        relations = relations.drop_duplicates(subset=['TERM_HEAD', 'TERM_TAIL', 'REL'])
        relations.reset_index(drop=True, inplace=True)
        utils.save_sheet(relations, metainfo_fp)
        metainfo = relations
    else:
        metainfo = utils.load_sheet(metainfo_fp)
    return metainfo

def load_cpubmed():
    src_fp = os.path.join(cfg.CPUBMED_SRC, "CPubMed-KGv1_1.txt")
    try:
        data = pd.read_csv(src_fp, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    except:
        with open(src_fp, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
        lines = [l.split('\t') for l in lines]
        lines = [l for l in lines if len(l) == 3]
        lines = ['\t'.join(l) for l in lines]
        with open(src_fp, 'w', encoding='utf-8') as writer:
            writer.writelines(lines)
        data = pd.read_csv(src_fp, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    data.columns = ['HEAD', 'REL', 'TAIL']
    head_ents, head_types, tail_ents, tail_types = [], [], [], []
    for i in data['HEAD'].tolist():
        try:    res  = i.split('@@')
        except: res  = [np.nan, np.nan]
        if len(res) != 2: res = [np.nan, np.nan]
        head_ents.append(res[0])
        head_types.append(res[1])
    for i in data['TAIL'].tolist():
        try:    res  = i.split('@@')
        except: res  = [np.nan, np.nan]
        if len(res) != 2: res = [np.nan, np.nan]
        tail_ents.append(res[0])
        tail_types.append(res[1])
    data['HEAD_ENT'], data['HEAD_TYPE']  = head_ents, head_types
    data['TAIL_ENT'], data['TAIL_TYPE']  = tail_ents, tail_types
    data = data.drop(columns=['HEAD', 'TAIL']).dropna()
    return data

def load_filtered_cpubmed_relations():
    relations_fp = os.path.join(cfg.CPUBMED, f'relations.csv')
    if not os.path.exists(relations_fp):
        all_relations = load_cpubmed()
        seeds = set(load_topic_words())
        relations = all_relations[(all_relations['HEAD_ENT'].isin(seeds)) & (all_relations['TAIL_ENT'].isin(seeds))]
        relations = relations.loc[relations['HEAD_ENT'] != relations['TAIL_ENT']]
        utils.save_sheet(relations, relations_fp)
    else:
        relations = utils.load_sheet(relations_fp)
    return relations

def clean_medqa_question_(question_jsonl: dict) -> dict:
    for k in cfg.MEDQA_KEYS:
        assert k in question_jsonl, f"Key `{k} is illegal.`"
    assert len(question_jsonl.keys()) == len(cfg.MEDQA_KEYS)
    assert len(question_jsonl['options']) == 4
    assert question_jsonl['answer_idx'] in question_jsonl['options']
    assert question_jsonl['options'][question_jsonl['answer_idx']] == question_jsonl['answer']
    assert '（    ）' in question_jsonl['question']
    question_jsonl['question'] = re.sub('[0-9]．', '', question_jsonl['question'])
    return question_jsonl

def load_medqa() -> pd.DataFrame:
    combined_path = os.path.join(cfg.MEDQA_EXT, 'combined.tsv')
    if os.path.exists(combined_path):
        data = utils.load_sheet(combined_path, converters={'options': utils.convert_list_str_to_list})
    else:
        source = []
        for s in ['dev', 'test', 'train']:
            fp      = os.path.join(cfg.MEDQA_SRC, f"{s}.jsonl")
            source += utils.load_jsonl(fp)
        cleaned_data = []
        for q in source:
            try:    cleaned_data.append(clean_medqa_question_(q))
            except: continue
        data = {k: [] for k in cfg.MEDQA_KEYS}
        for k in cfg.MEDQA_KEYS:
            data[k] = [d[k] for d in cleaned_data]
        data = pd.DataFrame.from_dict(data)
        data = data[cfg.MEDQA_FILTERED_KEYS]
        data = data.rename(columns={'answer_idx': 'answer'})
        data['answer']   = [item.strip() for item in data['answer']]
        data['question'] = [item.strip() for item in data['question']]
        for i in range(len(data['question'])):
            if not data['question'][i].endswith("。"):
                data['question'][i] += "。"
        data = data[data['answer'].str.contains('A|B|C|D|E', na=False)]
        data['question'] = data['question'].astype('str')
        data['answer'] = data['answer'].astype('str')
        utils.save_sheet(data, combined_path)
    return data

def preload_mlecqa_():
    """
    Combine the dev, test and train set of each category into a unified file.
    1. Only keep the qtext, options, and answer properties from the original
        json files
    2. Replace the chinese white spaces with the english ones
    """
    for cat in cfg.MLECQA_CATEGORY:
        data_dump = []
        for split in ['dev', 'test', 'train']:
            fn = f"{cat}_{split}.json"
            fp = os.path.join(cfg.MLECQA_SRC, fn)
            data = utils.load_json(fp)
            data = [{k: v.replace('　', '  ') if '　' in v else v for k, v in d.items()} for d in data]
            data = [utils.filter_dict_keys_(dic) for dic in data]
            data_dump += data
        dst_path = os.path.join(cfg.MLECQA_EXT, f"{cat}.jsonl")
        utils.save_jsonl(data_dump, dst=dst_path)

def load_mlecqa() -> pd.DataFrame:
    """
    Basic preprocessing and loading the dataset. To keep the format
    aligned across all samples, we:
        1. Add an chinese period "。" to the question part of each sample
        2. Remove the additional white space and line break of each
            sample for both its answer and question
    """
    combined_path = os.path.join(cfg.MLECQA_EXT, 'combined.tsv')
    if os.path.exists(combined_path):
        data = utils.load_sheet(combined_path, converters={'options': utils.convert_list_str_to_list})
    else:
        jsonls = []
        for cat in cfg.MLECQA_CATEGORY:
            fn = f"{cat}.jsonl"
            fp = os.path.join(cfg.MLECQA_EXT, fn)
            if not os.path.exists(fp): preload_mlecqa_()
            jsonls += utils.load_jsonl(fp)
        data = {k: [] for k in cfg.MLECQA_FILTERED_KEYS}
        for k in cfg.MLECQA_FILTERED_KEYS:
            data[k] = [jl[k] for jl in jsonls]
        valid_index = []
        for i, (opt, ans) in enumerate(zip(data['options'], data['answer'])):
            if ans in opt:
                valid_index.append(i)
        for k in cfg.MLECQA_FILTERED_KEYS:
            data[k] = [data[k][i] for i in valid_index]
        data['answer'] = [item.strip() for item in data['answer']]
        data['qtext']  = [item.strip() for item in data['qtext']]
        for i in range(len(data['qtext'])):
            if not data['qtext'][i].endswith("。"):
                data['qtext'][i] += "。"
        data = pd.DataFrame.from_dict(data)
        data = data.rename(columns={'qtext': 'question'})
        data = data[data['answer'].str.contains('A|B|C|D|E', na=False)]
        data['question'] = data['question'].astype('str')
        data['answer']   = data['answer'].astype('str')
        utils.save_sheet(data, combined_path)
    return data

def preprocess_mlecqa_(data:pd.DataFrame, original_q_maxlen=20, qa_minlen=10, qa_maxlen=30,
        q_minlen=5, a_maxlen=10) -> pd.DataFrame:
    # Can also be used in MedQA. 
    data = data.drop_duplicates(subset=['answer', 'question'], ignore_index=True)
    options = data['options'].tolist()
    answers = data['answer'].tolist()
    answers = [opt[a] for a, opt in zip(answers, options)]
    questions = [q.strip() for q in data['question'].tolist()]
    contexts  = ['。'.join(q.split("。")[:-2]) for q in questions]
    questions = [q.split("。")[-2] for q in questions]
    questions = [i.replace('（    ）', '') for i in questions]
    cleaned_contexts, cleaned_questions = [], []
    for c, q in zip(contexts, questions):
        if len(q) > original_q_maxlen:
            q_sents   = q.split("，")
            q_sents   = [i for i in q_sents if i != ""]
            con_sents, q_sent = q_sents[:-1], q_sents[-1]
            con_sents = '，'.join(con_sents)
            con_sents = con_sents + '，' if len(con_sents) > 0 else con_sents
            cleaned_questions.append(q_sent)
            context   = f"{c + '。' if len(c) != 0 else c}{con_sents}"
            context   = context.replace("。，", "。")
            cleaned_contexts.append(context)
        else:
            cleaned_contexts.append(c + '。' if len(c) != 0 else c)
            cleaned_questions.append(q)
    data.loc[:, 'answer']       = answers
    data.loc[:, 'context']      = cleaned_contexts
    data.loc[:, 'question']     = cleaned_questions
    data.loc[:, 'wrong_answer'] = [[i for i in list(op.values()) if i != a] for a, op in zip(answers, options)]
    data.loc[:, 'all_answer']   = [''.join(list(op.values())) for op in options]
    data = data[~data['all_answer'].str.contains("|".join(cfg.MLECQA_EXCLUDE_FROM_ANSWERS))]
    data = data[~data['answer'].str.contains("|".join(cfg.MLECQA_EXCLUDE_FROM_ANSWERS))]
    data = data[['context', 'question', 'answer', 'wrong_answer']]
    data.loc[:, 'answer']   = [item.strip() for item in data['answer'].tolist()]
    data.loc[:, 'question'] = [item.strip() for item in data['question'].tolist()]
    data.loc[:, 'question'] = utils.remove_question_mark_(data['question'].tolist())
    data.loc[:, 'context']  = [item.strip() for item in data['context'].tolist()]
    data = data[~data['question'].str.contains("|".join(cfg.MLECQA_EXCLUDE_FROM_QUESTIONS))]
    answers, questions = data['answer'].tolist(), data['question'].tolist()
    data.loc[:, 'qa_length']  = [len(f"{answers[i]}{questions[i]}") for i in range(len(data))]
    data.loc[:, 'q_length']   = [len(f"{questions[i]}") for i in range(len(data))]
    data.loc[:, 'a_length']   = [len(f"{answers[i]}") for i in range(len(data))]
    data = data[
        (data['qa_length'] > qa_minlen) & 
        (data['qa_length'] <= qa_maxlen) & 
        (data['q_length'] > q_minlen) & 
        (data['a_length'] <= a_maxlen)]
    for k, d in zip(cfg.MLECQA_AMBIGUOUS_PAIRS['keep'], cfg.MLECQA_AMBIGUOUS_PAIRS['drop']):
        mask = (data['question'].str.contains(d)) & (~data['question'].str.contains(k))
        data = data[~mask]
    data = data[['context', 'question', 'answer', 'wrong_answer']]
    data = data.reset_index(drop=True)
    return data

def load_filtered_mlecqa_qa(data: pd.DataFrame, folder=cfg.MLECQA, save=True):
    # Can also be used in MedQA.
    fp = os.path.join(folder, f"qa.tsv")
    if not os.path.exists(fp):
        words = set(load_topic_words())
        pattern = '|'.join(words)
        if 'context' in data:
            contexts  = data['context'].tolist()
            questions = data['question'].tolist()
            data['context+question'] = [f"{contexts[i]}{questions[i]}" for i in range(len(data))]
            data = data[data['context+question'].str.contains(pattern, na=False)]
        else:
            data = data[data['question'].str.contains(pattern, na=False)]
        data = data.drop(columns=['context+question'])
        data = data.reset_index(drop=True)
        if save: utils.save_sheet(data, fp)
    else:
        data = utils.load_sheet(fp)
    return data

def filter_by_duplication(
        file_1, file_2,
        subset_columns=['context', 'answer', 'question'],
        keep='last'):
    """
    Remember to exclude the columns 'label' in the final output.
    Keep the samples in the second file and exclude them from the first one.
    """
    file_1['label'] = ["file_1" for _ in range(len(file_1))]
    file_2['label'] = ["file_2" for _ in range(len(file_2))]
    file_concat = pd.concat([file_1, file_2], ignore_index=True)
    file_concat = file_concat.drop_duplicates(
        subset=subset_columns,
        keep=keep, ignore_index=True)
    file_1 = file_concat[file_concat['label'] == "file_1"]
    file_2 = file_concat[file_concat['label'] == "file_2"]
    file_1 = file_1.drop(columns=['label'])
    file_2 = file_2.drop(columns=['label'])
    return file_1, file_2

def main():
    load_filtered_bios_metainfo()
    load_filtered_cpubmed_relations()
    medqa  = load_medqa()
    medqa  = preprocess_mlecqa_(medqa)
    utils.save_sheet(medqa, os.path.join(cfg.MEDQA_EXT, "qa.tsv"))
    medqa  = load_filtered_mlecqa_qa(medqa, folder=cfg.MEDQA, save=True)
    mlecqa = load_mlecqa()
    mlecqa = preprocess_mlecqa_(mlecqa)
    utils.save_sheet(mlecqa, os.path.join(cfg.MLECQA_EXT, "qa.tsv"))
    mlecqa = load_filtered_mlecqa_qa(mlecqa, folder=cfg.MLECQA, save=True)
    mlecqa, medqa = filter_by_duplication(mlecqa, medqa)
    utils.save_sheet(medqa,  fp=os.path.join(cfg.MEDQA,  "qa.tsv"))
    utils.save_sheet(mlecqa, fp=os.path.join(cfg.MLECQA, "qa.tsv"))

if __name__ == "__main__":
    main()