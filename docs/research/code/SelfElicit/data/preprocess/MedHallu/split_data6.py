import pandas as pd
from split_utils import *
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "claim_ext_慢病幻觉数据集20240402")
    os.makedirs(cache_path, exist_ok=True)

    samples = []
    data_path = os.path.join(data_path, "claim_ext_慢病幻觉数据集20240402.xlsx")
    df = pd.read_excel(data_path, sheet_name='Sheet1')
    for _, item in tqdm(df.iterrows(), total=len(df), desc="claim_ext_慢病幻觉数据集20240402"):
        question = item['评测问题']
        if not isinstance(question, str):
            continue

        answer = item["draft"]
        answer = answer[answer.index('result:') + 7:] if 'result:' in answer else answer
        answer = answer[answer.index('answer:') + 7:] if 'answer:' in answer else answer

        reason = item['原因']
        reason = reason if isinstance(reason, str) else ""

        claims = item['claims']
        if not isinstance(claims, str):
            continue
        claims = claims.replace('\n\n', '\n').split('\n')
        claims = drop_short(claims)
        claims = [re.sub(r'\d.\s+', "", c.strip()) for c in claims]
        sentences = get_sentences(answer)
        sentences = drop_short(sentences)

        _from = item['draft来源'].replace('\n', '')
        if _from == '慢病V0.11':
            continue

        label = item['错误类型'] != 1
        correctness_reason = '\n'.join(filter(lambda line: '专业性' in line, reason.splitlines()))

        if not label:
            if not reason:
                continue
            else:
                claims_label = label_sentences_with_gpt(claims, correctness_reason, os.path.join(cache_path, f'{len(samples)}-err.txt'))
                sentences_label = label_sentences_with_gpt(sentences, correctness_reason, os.path.join(cache_path, f'{len(samples)}-sent.txt'))
        else:
            claims_label = [True] * len(claims)
            sentences_label = [True] * len(sentences)

        sample = {
            "question": question,
            "answer": answer,
            "from": f"慢病V0.11-{_from}",
            "score": int(item['专业性']),
            "label": label,
            "reason": reason,
            "claims": claims,
            "claim_labels": claims_label,
            "sentences": sentences,
            "sentence_labels": sentences_label
        }

        samples.append(sample)

    check_keys(samples)
    write(samples, "慢病幻觉数据集20240402.jsonl")
