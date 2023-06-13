import utils
import os
import pandas as pd
import config as cfg

def construct_dataset():
    bios = pd.DataFrame()
    bios["statement"]     = utils.load_sheet(os.path.join(cfg.BIOS, f"statements.tsv"))["statement"].tolist()
    bios["statement-neg"] = utils.load_sheet(os.path.join(cfg.BIOS, f"statements-neg.tsv"))["statement-neg"].tolist()
    bios["statement-rep"] = utils.load_sheet(os.path.join(cfg.BIOS, f"statements-rep.tsv"))["statement-rep"].tolist()
    bios["question-tf"]   = utils.load_sheet(os.path.join(cfg.BIOS, f"statements-qg.tsv"))["tf"].tolist()
    bios["question-open"] = utils.load_sheet(os.path.join(cfg.BIOS, f"statements-qg.tsv"))["open"].tolist()
    bios["context"] = ["" for _ in range(len(bios))]

    cpubmed = pd.DataFrame()
    cpubmed["statement"]     = utils.load_sheet(os.path.join(cfg.CPUBMED, f"statements.tsv"))["statement"].tolist()
    cpubmed["statement-neg"] = utils.load_sheet(os.path.join(cfg.CPUBMED, f"statements-neg.tsv"))["statement-neg"].tolist()
    cpubmed["statement-rep"] = utils.load_sheet(os.path.join(cfg.CPUBMED, f"statements-rep.tsv"))["statement-rep"].tolist()
    cpubmed["question-tf"]   = utils.load_sheet(os.path.join(cfg.CPUBMED, f"statements-qg.tsv"))["tf"].tolist()
    cpubmed["question-open"] = utils.load_sheet(os.path.join(cfg.CPUBMED, f"statements-qg.tsv"))["open"].tolist()
    cpubmed["context"] = ["" for _ in range(len(cpubmed))]

    medqa = utils.load_sheet(os.path.join(cfg.MEDQA, f"statements.tsv"))
    medqa_context = medqa['context'].tolist()
    for i in range(len(medqa_context)):
        if type(medqa_context[i]) != str:
            medqa_context[i] = ""
    medqa_statement = medqa['statement'].tolist()
    medqa_statement = [str(medqa_context[i]) + str(medqa_statement[i]) for i in range(len(medqa))]
    medqa["statement-neg"] = utils.load_sheet(os.path.join(cfg.MEDQA, f"statements-neg.tsv"))["statement-neg"].tolist()
    medqa["statement-rep"] = utils.load_sheet(os.path.join(cfg.MEDQA, f"statements-rep.tsv"))["statement-rep"].tolist()
    medqa["question-tf"]   = utils.load_sheet(os.path.join(cfg.MEDQA, f"statements-qg.tsv"))["tf"].tolist()
    medqa["question-open"] = utils.load_sheet(os.path.join(cfg.MEDQA, f"statements-qg.tsv"))["open"].tolist()
    medqa = medqa[["context", "statement", "statement-neg", "statement-rep", "question-tf", "question-open"]]

    mlecqa = utils.load_sheet(os.path.join(cfg.MLECQA, f"statements.tsv"))
    mlecqa_context = mlecqa['context'].tolist()
    for i in range(len(mlecqa_context)):
        if type(mlecqa_context[i]) != str:
            mlecqa_context[i] = ""
    mlecqa_statement = mlecqa['statement'].tolist()
    mlecqa_statement = [str(mlecqa_context[i]) + str(mlecqa_statement[i]) for i in range(len(mlecqa))]
    mlecqa["statement-neg"] = utils.load_sheet(os.path.join(cfg.MLECQA, f"statements-neg.tsv"))["statement-neg"].tolist()
    mlecqa["statement-rep"] = utils.load_sheet(os.path.join(cfg.MLECQA, f"statements-rep.tsv"))["statement-rep"].tolist()
    mlecqa["question-tf"]   = utils.load_sheet(os.path.join(cfg.MLECQA, f"statements-qg.tsv"))["tf"].tolist()
    mlecqa["question-open"] = utils.load_sheet(os.path.join(cfg.MLECQA, f"statements-qg.tsv"))["open"].tolist()
    mlecqa = mlecqa[["context", "statement", "statement-neg", "statement-rep", "question-tf", "question-open"]]

    dataset = pd.concat([bios, cpubmed, medqa, mlecqa], axis=0)
    sources =\
        ['bios'] * len(bios) + ['cpubmed'] * len(cpubmed) + \
        ['medqa'] * len(medqa) + ['mlecqa'] * len(mlecqa)
    dataset['source'] = sources
    dataset = dataset.dropna(subset=["statement", "statement-neg", "statement-rep"])
    utils.save_sheet(dataset, os.path.join(cfg.BENCHMARK, "statements.tsv"))

def get_dataset_for_annotation():
    data     = utils.load_sheet(os.path.join(cfg.BENCHMARK, "statements.tsv"))
    annotation_df = pd.DataFrame()
    contexts = data["context"].tolist()
    q_tfs    = data["question-tf"].tolist()
    q_opens  = data["question-open"].tolist()
    contexts = [c if isinstance(c, str) else "" for c in contexts]
    sources  = data["source"].tolist()
    answers  = data["statement"].tolist()
    a_tfs    = data["statement-neg"].tolist()
    a_opens  = data["statement-rep"].tolist()
    annotation_questions     = []
    annotation_answers       = []
    annotation_contexts      = []
    annotation_answers_wrong = []
    annotation_q_type        = []
    original_sources         = []
    original_indexs          = []
    for i in range(len(data)):
        q_tf = q_tfs[i]
        if isinstance(q_tf, str):
            annotation_q_type.append("tf")
            annotation_questions.append(q_tf)
            annotation_answers.append(answers[i])
            annotation_contexts.append(contexts[i])
            annotation_answers_wrong.append(a_tfs[i])
            original_sources.append(sources[i])
            original_indexs.append(i)
        q_op = q_opens[i]
        if isinstance(q_op, str):
            annotation_q_type.append("open")
            annotation_questions.append(q_op)
            annotation_answers.append(answers[i])
            annotation_contexts.append(contexts[i])
            annotation_answers_wrong.append(a_opens[i])
            original_sources.append(sources[i])
            original_indexs.append(i)
    annotation_df["index"]        = original_indexs
    annotation_df["q-type"]       = annotation_q_type
    annotation_df["source"]       = original_sources
    annotation_df["context"]      = annotation_contexts
    annotation_df["question"]     = annotation_questions
    annotation_df["answer"]       = annotation_answers
    annotation_df["answer-wrong"] = annotation_answers_wrong
    utils.save_sheet(annotation_df, os.path.join(cfg.BENCHMARK, "questions_original.tsv"))

def main():
    construct_dataset()
    get_dataset_for_annotation()

main()