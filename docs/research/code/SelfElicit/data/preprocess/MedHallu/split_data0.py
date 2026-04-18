import pandas as pd
from split_utils import *
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split/"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "V0.15评测问题记录")
    os.makedirs(cache_path, exist_ok=True)

    samples = []

    data_path = os.path.join(data_path, "New_claim_ext_V0.15_labeled.xlsx")
    df = pd.read_excel(data_path, sheet_name="Sheet1")

    for i, line in tqdm(df.iterrows(), total=len(df)):
        reason = line['错误点']
        reason = reason if isinstance(reason, str) else ""

        answer = line['draft']
        answer = answer[answer.index('result:') + 7:] if 'result:' in answer else answer
        answer = answer[answer.index('answer:') + 7:] if 'answer:' in answer else answer

        claims = split_answer_with_gpt(answer, os.path.join(cache_path, f'Sheet1-{i}.txt'))
        claims = drop_short(claims)
        claims = [c.replace("- ", '').replace("\n", '') for c in claims]
        sentences = get_sentences(answer)
        sentences = drop_short(sentences)

        label = line['知识错误'] == 0
        if not label:
            claims_label = label_sentences_with_gpt(claims, reason, os.path.join(cache_path, f'Sheet1-{i}-err.txt'))
            sentences_label = label_sentences_with_gpt(sentences, reason, os.path.join(cache_path, f'Sheet1-{i}-sent.txt'))
        else:
            claims_label = [True] * len(claims)
            sentences_label = [True] * len(sentences)

        sample = {
            "question": line['评测问题'],
            "answer": answer,
            "from": "慢病V0.15",
            "claims": claims,
            "claim_labels": claims_label,
            "sentences": sentences,
            "sentence_labels": sentences_label,
            "label": label,
            "reason": reason,
            "score": line['专业性']
        }
        samples.append(sample)

    check_keys(samples)
    write(samples, "V0.15评测问题记录.jsonl")
