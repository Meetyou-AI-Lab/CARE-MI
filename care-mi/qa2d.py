from instructions import *
from joblib import Parallel, delayed
from tqdm import tqdm
import utils
import os
import openai
import time
import argparse
import config as cfg
import pandas as pd
import numpy as np

os.environ['OPENAI_API_KEY']=cfg.OPENAI_API_KEY

def get_chat_input_(question: str, answer: str):
    return f"""
{QA2D_CHATGPT_USER_PROMPT}
Q:{question}
A:{answer}
D:"""

def qa2d_api_call_(question: str, answer: str):
    input = get_chat_input_(question, answer)
    try:
        completion=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": QA2D_CHATGPT_SYSTEM_PROMPT},
                {"role": "user",   "content": input}
            ],
        temperature=0,
        max_tokens=128
        )
        response = completion['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response = np.nan
    return response

def qa2d_async_(questions: list, answers: list, njobs=0):
    if njobs <= 0:
        njobs = os.cpu_count()
    respones = Parallel(n_jobs=njobs)(delayed(qa2d_api_call_)(q, a) for q, a in tqdm(zip(questions, answers)))
    return respones

def eval_failed_(data: pd.DataFrame, results: list, njobs: int):
    failed_idxs      = [i for i, r in enumerate(results) if type(r) == float]
    failed_samples   = data.filter(items=failed_idxs, axis=0)
    failed_questions = failed_samples['question'].tolist()
    failed_answers   = failed_samples['answer'].tolist()
    failed_results   = qa2d_async_(failed_questions, failed_answers, njobs)
    for i, res in zip(failed_idxs, failed_results):
        results[i] = res
    return results

def main(args):
    if   args.dataset == "MEDQA":
        folder = cfg.MEDQA
    elif args.dataset == "MLECQA":
        folder = cfg.MLECQA
    fp   = os.path.join(folder, "qa.tsv")
    data = utils.load_sheet(fp, converters={'wrong_answer': utils.convert_list_str_to_list})
    questions = data['question'].tolist()
    answers   = data['answer'].tolist()
    results   = [np.nan] * len(data)
    for i in range(len(data)):
        q, a = questions[i], answers[i]
        qa   = (q + a).strip('。')
        if q[-1] == "是"    or q[-1] == "为"    or q[-1] == "的"   or q[-1] == "于"\
        or q[-2:] == "属于" or q[-2:] == "可致" or q[-2:] == "选用" or q[-2:] == "首选" or q[-2:] == "选择":
            results[i] = qa + "。"
    results = eval_failed_(data, results, args.njobs)
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
    data["statement"] = results
    save_fp = os.path.join(folder, "statements.tsv")
    utils.save_sheet(data, save_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate declarative sentences using QA pairs.')
    parser.add_argument('--dataset',     type=str,   default="MEDQA", choices=["MEDQA", "MLECQA"])
    parser.add_argument('--njobs',       type=int,   default=15)
    parser.add_argument('--max_iter',    type=int,   default=3, help="Max iterations of failed samples evaluation.")
    parser.add_argument('--interval',    type=float, default=5.0, help="Time interval waited between each round.")
    parser.add_argument('--eval_failed', action='store_false', default=True)
    args = parser.parse_args()
    main(args)