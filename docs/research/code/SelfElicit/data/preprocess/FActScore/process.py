import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = './data/original_dataset/FActScore'
    output_dir = './data/dataset/FActScore/'
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if not file.endswith('.jsonl'):
            continue

        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        samples = []
        for i, line in enumerate(tqdm(lines)):
            legal = True
            data = json.loads(line)

            replaceName = lambda text: text.replace("He", data['topic']).replace("She", data['topic'])

            claims, claim_labels = [], []
            sentences, sentence_labels = [], []

            if not data['annotations']:
                print(f"Skip unlabelled item")
                legal = False
                continue

            for items in data['annotations']:
                if not items['is-relevant']:
                    print(f"Skip irrelevant sentence")
                    legal = False
                    continue

                sentences.append(replaceName(items['text']))

                flag = True
                for item in items['human-atomic-facts']:
                    claims.append(replaceName(item['text']))
                    if item['label'] == "NS":
                        claim_labels.append(0)
                        flag = False
                    else:
                        claim_labels.append(1)

                if flag:
                    sentence_labels.append(1)
                else:
                    sentence_labels.append(0)

            assert len(claims) == len(claim_labels) and len(sentences) == len(sentence_labels)
            sample = {
                'question': data['input'].replace("Question: ", ""),
                'answer': data['output'],
                'from': "FActScore-" + file.replace('.jsonl', ''),
                'claims': claims,
                "claim_labels": claim_labels,
                "sentences": sentences,
                "sentence_labels": sentence_labels,
                "label": all(claim_labels)
            }
            if legal:
                samples.append(sample)

        with open(os.path.join(output_dir, file), 'w', encoding='utf-8') as fp:
            for sample in samples:
                fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
