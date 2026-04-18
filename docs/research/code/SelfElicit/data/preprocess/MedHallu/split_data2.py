import pandas as pd
from split_utils import *
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "qwen_14B_opinion_extract_v0.16")
    os.makedirs(cache_path, exist_ok=True)

    samples = []

    data_path = os.path.join(data_path, "qwen_14B_opinion_extract_v0.16.xlsx")
    df = pd.read_excel(data_path, sheet_name="qwen_14B_opinion_extract_v0.16")
    for i, line in tqdm(df.iterrows(), total=len(df), desc="qwen_14B_opinion_extract_v0.16"):
        reason = line['原因']
        reason = reason if isinstance(reason, str) else ""

        answer = line["回答"]
        if not isinstance(answer, str):
            continue
        answer = answer[:answer.index('注：参考自')] if '注：参考自' in answer else answer
        answer = answer.strip()
        claims = split_answer_with_gpt(answer, os.path.join(cache_path, f'{i}.txt'))
        claims = drop_short(claims)
        claims = [c.replace("- ", '').replace("\n", '') for c in claims]
        sentences = get_sentences(answer)
        sentences = drop_short(sentences)

        label = line['知识错误'] == 0
        if not label:
            claims_label = label_sentences_with_gpt(claims, reason, os.path.join(cache_path, f'{i}-claim.txt'))
            sentences_label = label_sentences_with_gpt(sentences, reason, os.path.join(cache_path, f'{i}-sent.txt'))
        else:
            claims_label = [True] * len(claims)
            sentences_label = [True] * len(sentences)

        sample = {
            "question": line['问题'],
            "answer": answer,
            "from": "慢病V0.16",
            "claims": claims,
            "claim_labels": claims_label,
            "sentences": sentences,
            "sentence_labels": sentences_label,
            "label": label,
            "reason": reason,
            "score": line['专业']
        }
        samples.append(sample)

    check_keys(samples)
    write(samples, "qwen_14B_opinion_extract_v0.16.jsonl")
