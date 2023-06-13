import os
import argparse
import utils
import config as cfg

def CPUBMED_STATEMENT_TEMPLATES_(rel: str, head: str, tail: str) -> str:
    if rel == '发病部位' or rel == '就诊科室' or rel == '转移部位' \
        or rel == '发病年龄' or rel == '发病率' or rel == '死亡率' \
        or rel == '发病性别倾向' or rel == '预后生存率' or rel == '多发地区' \
        or rel == '多发群体' or rel == '发病机制' or rel == '外侵部位' \
        or rel == '多发季节':
        s = f'{head}的{rel}是{tail}'
    elif rel == '预防':
        s = f'{tail}有助于{rel}{head}'
    elif rel == '相关（导致）':
        s = f'{head}可能会导致{tail}'
    elif rel == '风险评估因素' or rel == '高危因素' or rel =='病因' or rel == '病史' \
        or rel == '辅助治疗' or rel == '药物治疗' or rel == '手术治疗' or rel == '放射治疗' \
        or rel == '病理分型' or rel == '临床表现' or rel == '并发症' or rel == '治疗后症状' \
        or rel == '化疗' or rel == '病理生理' or rel == '阶段' or rel == '遗传因素':
        s = f'{head}的{rel}包含{tail}'
    elif rel == '内窥镜检查' or rel == '影像学检查' or rel == '实验室检查' or rel == '组织学检查' \
        or rel == '筛查' or rel == '辅助检查':
        s = f'{head}可通过{tail}进行{rel}'
    elif rel == '鉴别诊断':
        s = f'{head}可以{rel}{tail}'
    elif rel == '传播途径':
        s = f'{tail}是{head}的{rel}'
    elif rel == '相关（转化）':
        s = f'{head}可能会转化为{tail}'
    elif rel == '相关（症状）':
        s = f'{head}的相关症状包含{tail}'
    elif rel == '预后状况':
        s = f'{head}的{rel}{tail}'
    else:
        print(rel)
        raise NotImplementedError
    return s

def CPUBMED_construct_statements_():
    fp   = os.path.join(cfg.CPUBMED, "relations.csv")
    data = utils.load_sheet(fp)
    rels = data['REL'].tolist()
    heads = data['HEAD_ENT'].tolist()
    tails = data['TAIL_ENT'].tolist()
    statements = [CPUBMED_STATEMENT_TEMPLATES_(rels[i], heads[i], tails[i]) for i in range(len(data))]
    data[f'statement'] = statements
    save_fp = os.path.join(cfg.CPUBMED, "statements.tsv")
    utils.save_sheet(data, save_fp)

def BIOS_construct_statements_():
    fp   = os.path.join(cfg.BIOS, f'metainfo.tsv')
    data = utils.load_sheet(fp)
    statements = []
    for item in data.iterrows():
        item = item[1]
        rel = item['REL']
        head = item['TERM_HEAD'].title()
        tail = item['TERM_TAIL']
        if   rel == '禁忌用药':   statements.append(f"{head}是{tail}的{rel}")
        elif rel == '有相互作用': statements.append(f"{head}和{tail}有{rel}")
        elif rel == '有不良反应（反向）':
            rel = '不良反应'
            statements.append(f"{rel}{head}可被{tail}产生")
        elif rel == '可导致（反向）': statements.append(f"{head}可由{tail}导致")
        elif rel == '可诊断（反向）': statements.append(f"{head}可由{tail}诊断")
        elif rel == '可治疗（反向）': statements.append(f"{head}可由{tail}治疗")
        elif rel == '是一种（反向）': statements.append(f"{head}包含{tail}")
        else: statements.append(f"{head}{rel}{tail}")
    data['statement'] = statements
    save_fp = os.path.join(cfg.BIOS, f'statements.tsv')
    utils.save_sheet(data, fp=save_fp)
    return data

def main(args):
    try:
        if   args.dataset == 'BIOS':
            BIOS_construct_statements_()
        elif args.dataset == 'CPUBMED':
            CPUBMED_construct_statements_()
    except:
        print(f"Doing preprocessing for {args.dataset} now...")
        os.system(f"python {os.path.join(cfg.CODE, 'preprocess.py')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Building statements from KG datasets.')
    parser.add_argument('--dataset', type=str, default='CPUBMED', choices=["BIOS", "CPUBMED"])
    args = parser.parse_args()
    main(args)
