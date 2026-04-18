import pandas as pd
from split_utils import *
from tqdm import tqdm

cache_path = "./data/cache/MedHallu/split"
data_path = "./data/original_dataset/MedHallu"

if __name__ == '__main__':
    cache_path = os.path.join(cache_path, "claims_check_data_v3.2")
    os.makedirs(cache_path, exist_ok=True)

    data_path = os.path.join(data_path, "claims_check_data_v3.2.xlsx")
    df_dict = pd.read_excel(data_path, sheet_name=["用药建议", "v0.11_sample"])

    samples = []
    drop_count = 0
    for df in [df_dict['用药建议'], df_dict['v0.11_sample']]:
        df = df.groupby("评测问题")
        for question, group in tqdm(df, total=len(df), desc="claims_check_data_v3.2"):
            answer = group['回答'].tolist()[0]
            if 'trace_id' in answer:
                answer = answer[answer.index('result:') + 7:]

            claims = group['claims 人工标定'].tolist()
            claims = drop_short(claims)
            claim_labels = [l != 1 for l in group['正确性'].tolist()]
            assert len(claims) == len(claim_labels), (claims, claim_labels)

            sentences = get_sentences(answer)
            sentences = drop_short(sentences)
            reason = "。".join([f"\"{claims[i]}\"有误" for i in range(len(claims)) if not claim_labels[i]])
            if not all(claim_labels):
                sentences_label = label_sentences_with_gpt(sentences, reason, os.path.join(cache_path, f'{len(samples)}-sent.txt'))
            else:
                sentences_label = [True] * len(sentences)

            sample = {
                "question": question,
                "answer": answer,
                "claims": claims,
                "claim_labels": claim_labels,
                "label": all(claim_labels),
                "from": "慢病V0.11",
                'sentences': sentences,
                "sentence_labels": sentences_label
            }
            samples.append(sample)

    check_keys(samples)
    write(samples, "claims_check_data_v3.2.jsonl")
