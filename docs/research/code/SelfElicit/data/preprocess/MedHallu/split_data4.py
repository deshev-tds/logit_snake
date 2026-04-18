import pandas as pd
from split_utils import *
import numpy as np
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "v017_拆分_claims_评测2")
    os.makedirs(cache_path, exist_ok=True)

    samples = []

    data_path = os.path.join(data_path, "v017_拆分_claims_评测2.xlsx")
    df = pd.read_excel(data_path, sheet_name="Sheet1")
    for i, line in tqdm(df.iterrows(), total=len(df), desc="v017_拆分_claims_评测2"):
        reason = line['原因']
        reason = reason if isinstance(reason, str) else ""

        answer = line['评测2 （0131版本 无知识增强）']
        answer = answer[answer.index('##回复') + 4:] if '##回复' in answer else answer
        answer = answer.strip()
        claims = split_answer_with_gpt(answer, os.path.join(cache_path, f'{i}.txt'))
        claims = drop_short(claims)
        claims = [c.replace("- ", '').replace("\n", '') for c in claims]
        sentences = get_sentences(answer)
        sentences = drop_short(sentences)

        label = line['知识错误'] != 1
        if not label:
            claims_label = label_sentences_with_gpt(claims, reason, os.path.join(cache_path, f'{i}-err.txt'))
            sentences_label = label_sentences_with_gpt(sentences, reason, os.path.join(cache_path, f'{i}-sent.txt'))
        else:
            claims_label = [True] * len(claims)
            sentences_label = [True] * len(sentences)

        sample = {
            "question": line['问题'],
            "answer": answer,
            "reason": reason,
            "from": "云诊室评测2",
            "claims": claims,
            "claim_labels": claims_label,
            "sentences": sentences,
            "sentence_labels": sentences_label,
            "label": label,
            "score": 2 if np.isnan(line['专业分']) else line['专业分']
        }
        samples.append(sample)

    check_keys(samples)
    write(samples, "v017_拆分_claims_评测2.jsonl")
