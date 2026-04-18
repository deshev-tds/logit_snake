import os
import re
import json
import nltk
from tqdm import tqdm
import itertools
from utils.llm import *

model_path = "deepseek"
dataset = "WikiBio"

io_path = "./saves/IO/WikiBio/qwen1_5_7b_chat-sentence/log.txt"
far_path = "./saves/FaR/WikiBio/qwen1_5_7b_chat-sentence/log.txt"
context_path = "./saves/IO/WikiBio/qwen1_5_7b_chat-sentence-context/log.txt"
cot_path = "./saves/CoT/WikiBio/qwen2_5_7b_chat-sentence-1/log.txt"
elicit_path = "./saves/thought/WikiBio/qwen1_5_7b_chat-sentence/log.txt"
paths = {
    'IO': io_path,
    'FaR': far_path,
    'context': context_path,
    'cot': cot_path,
    'elicit': elicit_path
}

blacklist = ["请选择", "正确", "错误", '无法', 'Response',
             'True', 'False', 'answer', 'Not Sure',
             "所以", '因此', '我', 'Therefore', 'Correct', 'Incorrect',
             'accurate', 'inaccurate', 'error', 'description', 'statement', 'explanation',
             'graph', '[', ']', 'default']


def get_io_log(file):
    with open(file, 'r', encoding='utf-8') as f:
        log = "".join(f.readlines())
        log = log.split("Total Cost")[0]
        log = re.split(r"\**\[\d+/\d+\]\**", log)[1:]

    claims, reflections = [], []
    for smp in log:
        items = re.split(r"(###Query|###Response)", smp)
        items = [s.strip() for s in items if "#Query" not in s and "#Response" not in s]
        items = list(filter(lambda s: len(s) > 0, items))
        claim, ref = items[0::2], items[1::2]

        claim = [re.findall(r"Description: (.*)", s)[0] for s in claim if "#" not in s]

        reflection = []
        for r in ref:
            temp = []
            for s in nltk.sent_tokenize(r):
                if any(_ in s for _ in blacklist) or len(s) < 5:
                    continue
                temp.append(s)
            reflection.append(temp)

        claims.append(claim)
        reflections.append(reflection)

    return claims, reflections


def get_elicit_log(file):
    with open(file, 'r', encoding='utf-8') as f:
        log = "".join(f.readlines())
        log = log.split("Total Cost")[0]
        log = re.split(r"\**\[\d+/\d+\]\**", log)[1:]

    claims, reflections = [], []
    for smp in log:
        items = re.split(r"(###Query|###Response)", smp)
        items = [s.strip() for s in items if "#Query" not in s and "#Response" not in s]
        items = [s for s in items if not any(s.startswith(x) for x in ['graph', '[', 'default'])]
        items = list(filter(lambda s: len(s) > 0, items))
        claim, ref = items[0::2], items[1::2]

        claim = [re.findall(r"Description: (.*)", s)[0] for s in claim if "#" not in s]

        reflection = []
        for r in ref:
            temp = []
            for s in nltk.sent_tokenize(r):
                if any(_.lower() in s.lower() for _ in blacklist) or len(s) < 5:
                    continue
                temp.append(s)
            reflection.append(temp)

        claims.append(claim)
        reflections.append(reflection)

    return claims, reflections


def get_far_log(file):
    with open(file, 'r', encoding='utf-8') as f:
        log = "".join(f.readlines())
        log = log.split("Total Cost")[0]
        log = re.split(r"\**\[\d+/\d+\]\**", log)[1:]

    claims, reflections = [], []
    for smp in log:
        items = re.split(r"(###Query|###Response)", smp)
        items = [s.strip() for s in items if "#Query" not in s and "#Response" not in s]
        items = list(filter(lambda s: len(s) > 0, items))
        assert len(items) % 6 == 0

        claim, reflection = items[0::6], items[1::6]

        claim = [s.split("\n")[1] for s in claim]
        reflection = [s.split("\n") for s in reflection]

        claims.append(claim)
        reflections.append(reflection)

    return claims, reflections


def get_cot_log(file):
    with open(file, 'r', encoding='utf-8') as f:
        log = "".join(f.readlines())
        log = log.split("Total Cost")[0]
        log = re.split(r"\**\[\d+/\d+\]\**", log)[1:]

    claims, reflections = [], []
    for smp in log:
        items = re.split(r"(###Query|###Response)", smp)
        items = [s.strip() for s in items if "#Query" not in s and "#Response" not in s]
        items = list(filter(lambda s: len(s) > 0, items))
        assert len(items) % 2 == 0

        claim, ref = items[0::2], items[1::2]

        claim = [s.split("\n")[0] for s in claim]
        reflection = []
        for r in ref:
            temp = []
            for s in nltk.sent_tokenize(r):
                if any(_.lower() in s.lower() for _ in blacklist) or len(s) < 5:
                    continue
                temp.append(s)
            reflection.append(temp)

        claims.append(claim)
        reflections.append(reflection)

    return claims, reflections


if __name__ == '__main__':
    # load data
    all_claims, all_reflections = {}, {}
    all_claims['IO'], all_reflections['IO'] = get_io_log(paths['IO'])
    all_claims['FaR'], all_reflections['FaR'] = get_far_log(paths['FaR'])
    all_claims['context'], all_reflections['context'] = get_io_log(paths['context'])
    all_claims['elicit'], all_reflections['elicit'] = get_elicit_log(paths['elicit'])
    all_claims['cot'], all_reflections['cot'] = get_cot_log(paths['cot'])

    # load model
    model = APIModel(None)

    x = 4
    x = 50 * x
    y = x + 50

    # check fact
    for k in all_reflections:
        file = f"./plot/quality/output/{dataset}/fact-{k}-{x}-{y}.txt"

        # 断点续传
        if not os.path.exists(file):
            with open(file, 'w', encoding='utf-8') as f:
                pass
        with open(file, 'r', encoding='utf-8') as f:
            cur_id = len(f.readlines())
            cur_id = cur_id + x

        loader = tqdm(zip(all_claims[k], all_reflections[k]), total=len(all_claims[k]), desc=k)
        for i, (claim, reflection) in enumerate(loader):
            if not cur_id <= i < y:
                continue

            print(i)
            model.verbose = True

            all_fact = []
            # assert len(claim) == len(reflection)
            for c, r in zip(claim, reflection):
                facts = []
                context = []
                for item in r:
                    if context:
                        query = '上文：' + "。".join(context) + "\n"
                    else:
                        query = ""
                    query += f"请判断以下句子的事实性，并使用**正确**或者**错误**来回答。请不要解释。\n"
                    query += f"句子：{item}"
                    _, response, _ = model.understand(query)
                    loader.set_description(f"{k}-{response}")
                    if '错误' in response or '不正确' in response:
                        fact = 0
                    elif '正确' in response:
                        fact = 1
                        context.append(item)
                    else:
                        fact = 0.5
                    facts.append(fact)

                all_fact.append(facts)

            with open(file, 'a', encoding='utf-8') as f:
                s = json.dumps(all_fact)
                f.write(f'{s}\n')
