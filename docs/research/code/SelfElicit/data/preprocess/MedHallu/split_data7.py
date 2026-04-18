import pandas as pd
from split_utils import *
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "银屑病0229")
    os.makedirs(cache_path, exist_ok=True)

    data_path = os.path.join(data_path, "银屑病0229.xlsx")
    df = pd.read_excel(data_path, sheet_name='Sheet1')

    samples = []

    for column in ['夸克', 'GPT4', 'GPT3.5', '云诊室预发']:
        for i in tqdm(range(len(df[column])), total=len(df), desc="银屑病0229-" + column):
            question = df['问题'].iloc[i]
            if not isinstance(question, str):
                continue

            answer = df[column].iloc[i]
            if not isinstance(answer, str):
                continue

            claims = split_answer_with_gpt(answer, os.path.join(cache_path, f'{column}-{i}.txt'))
            claims = drop_short(claims)
            claims = [c.replace("- ", '').replace("\n", '') for c in claims]
            sentences = get_sentences(answer)
            sentences = drop_short(sentences)

            reason = df[f"{column}-原因"].iloc[i]
            reason = reason if isinstance(reason, str) else ""

            label = bool(df[f"{column}-知识错误"].iloc[i] == 0)
            if not label:
                claims_label = label_sentences_with_gpt(claims, reason, os.path.join(cache_path, f'{column}-{i}-err.txt'))
                sentences_label = label_sentences_with_gpt(sentences, reason, os.path.join(cache_path, f'{column}-{i}-sent.txt'))
            else:
                claims_label = [True] * len(claims)
                sentences_label = [True] * len(sentences)

            sample = {
                'question': question,
                'answer': answer,
                'claims': claims,
                'score': int(df[f"{column}-专业性"].iloc[i]),
                'label': label,
                'reason': reason,
                'from': f"慢病V0.18-{column}",
                'sentences': sentences,
                "claim_labels": claims_label,
                "sentence_labels": sentences_label
            }
            samples.append(sample)

    check_keys(samples)
    write(samples, "银屑病0229.jsonl")
