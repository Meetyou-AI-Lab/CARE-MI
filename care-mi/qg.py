import os
import re
import utils
import torch
import argparse
import config as cfg
import pandas as pd
from instructions import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda:0')
model_path="your/model/path" # Specify your model path here
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def preprocess(text: str) -> str:
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\t", "\t")

def ChatYuan_answer(text: str, sample=True, top_p=1, temperature=0.1) -> str:
    text = preprocess(text)
    encoding = tokenizer(
        text=[text],
        truncation=True,
        padding=True,
        max_length=768, 
        return_tensors="pt").to(device) 
    if not sample:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=512,
            num_beams=1,
            length_penalty=0.6)
    else:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=512,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])

def generate_tf_question(data: pd.DataFrame) -> pd.DataFrame:
    statements = data['statement'].tolist()
    if "context" in data:
        contexts = data['context'].tolist()
    else:
        contexts = [""] * len(data)
    questions  = []
    for statement, context in zip(statements, contexts):
        if isinstance(context, float):
            context = ""
        statement = context + statement
        prefix   = QG_CHATYUAN_TF_PROMPT['pre']
        postfix  = QG_CHATYUAN_TF_PROMPT['post']
        input    = f"{prefix}{statement}{postfix}"
        response = ChatYuan_answer(f"用户：{input}\n小元：")
        questions.append(response)
    data.loc[:, "tf"] = questions
    return data

def find_last_match(chars: str, string: str) -> int:
    match = re.finditer(chars, string)
    last_match = None
    for m in match:
        last_match = m
    if last_match is not None:
        return last_match.end() - 1
    else:
        return -1

def generate_open_question(data: pd.DataFrame) -> pd.DataFrame:
    statements = data['statement'].tolist()
    if "context" in data:
        contexts = data['context'].tolist()
    else:
        contexts = [""] * len(data)
    questions  = []
    for statement, context in zip(statements, contexts):
        if isinstance(context, float):
            context = ""
        statement = context + statement
        index = find_last_match("是|为", statement)
        if index != -1:
            response = f"{statement[:index + 1]}什么？"
        else:
            response = ""
        questions.append(response)
    data.loc[:, "open"] = questions
    return data

def main(args):
    if   args.dataset == "BIOS":
        folder = cfg.BIOS
    elif args.dataset == "CPUBMED":
        folder = cfg.CPUBMED
    elif args.dataset == "MEDQA":
        folder = cfg.MEDQA
    elif args.dataset == "MLECQA":
        folder = cfg.MLECQA
    fp   = os.path.join(folder, "statements.tsv")
    data = utils.load_sheet(fp)
    if args.dataset == "BIOS" or "CPUBMED":
        data = generate_tf_question(data)
        data.loc[:, "open"] = [""] * len(data)
    else:
        data = generate_tf_question(data)
        data = generate_open_question(data)
    utils.save_sheet(data, os.path.join(folder, f"statments-qg.tsv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate questions from declarative sentences.')
    parser.add_argument('--dataset', type=str, default="BIOS", choices=["MEDQA", "MLECQA", "BIOS", "CPUBMED"])
    args = parser.parse_args()
    main(args)