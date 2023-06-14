import config as cfg
import utils
import os
import re
import time
import argparse
import openai
import pandas as pd
import numpy as np
from instructions import *
from tqdm import tqdm
from joblib import Parallel, delayed

os.environ['OPENAI_API_KEY']=cfg.OPENAI_API_KEY

def get_chat_input_(statement: str) -> str:
    return f"""
{D2FD_CHATGPT_USER_PROMPT}
D:{statement}
N:"""

def negation_api_call_(statement: str) -> str:
    input = get_chat_input_(statement)
    try:
        completion=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": D2FD_CHATGPT_SYSTEM_PROMPT},
                {"role": "user",   "content": input}
            ],
        temperature=0
        )
        response = completion['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response = np.nan
    return response

def negation_async_(statements: list, njobs=0) -> list:
    if njobs <= 0:
        njobs = os.cpu_count()
    respones = Parallel(n_jobs=njobs)(delayed(negation_api_call_)(s) for s in tqdm(statements))
    return respones

def rule_based_negation_(statement: str) -> str:
    false_statement = np.nan
    if "最可能" in statement and "最可能的" not in statement:
        false_statement = statement.replace("最可能", "不可能")
    elif "应是" in statement or "应为" in statement:
        for match in re.finditer("|".join(["应是", "应为"]), statement):
            last_match = match
        start = last_match.start()
        false_statement = statement[:start] + "不" + statement[start:]
    else:
        try:
            for match in re.finditer("|".join(["应", "为", "是", "可"]), statement):
                last_match = match
            start = last_match.start()
            false_statement = statement[:start] + "不" + statement[start:]
        except:
            print(f"Error with `{statement}`.")
            false_statement = np.nan
    return false_statement

def batch_rule_based_negation_(statements: list) -> list:
    return [rule_based_negation_(s) for s in statements]

def eval_failed_(data: pd.DataFrame, results: list, njobs: int):
    failed_idxs = [i for i, r in enumerate(results) if type(r) == float]
    failed_samples = data.filter(items=failed_idxs, axis=0)
    failed_statements = failed_samples['statement'].tolist()
    failed_results = negation_async_(failed_statements, njobs)
    for i, res in zip(failed_idxs, failed_results):
        results[i] = res
    return results

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
    statements = data["statement"].tolist()
    results    = batch_rule_based_negation_(statements)
    results    = eval_failed_(data, results, args.njobs)
    if args.eval_failed == 1 and args.max_iter != 0:
        max_iter    = args.max_iter
        failed_idxs = [i for i, r in enumerate(results) if type(r) == float]
        while len(failed_idxs) != 0 and max_iter != 0:
            print(f"Failed sampels evaluation:")
            if max_iter != args.max_iter:
                print(f"Wait {args.interval} seconds...")
                time.sleep(args.interval)
            results = eval_failed_(data, results, args.njobs)
            max_iter -= 1
    data["statement-neg"] = results
    save_fp = os.path.join(folder, "statements-neg.tsv")
    utils.save_sheet(data, save_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate negated sentences from declarative sentences.')
    parser.add_argument('--dataset',     type=str,   default="BIOS", choices=["MEDQA", "MLECQA", "BIOS", "CPUBMED"])
    parser.add_argument('--njobs',       type=int,   default=15)
    parser.add_argument('--max_iter',    type=int,   default=3, help="Max iterations of failed samples evaluation.")
    parser.add_argument('--interval',    type=float, default=5.0, help="Time interval waited between each round.")
    parser.add_argument('--eval_failed', action='store_false', default=True)
    args = parser.parse_args()
    main(args)