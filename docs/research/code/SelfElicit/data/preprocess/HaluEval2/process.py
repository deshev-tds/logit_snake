import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = './data/original_dataset/HaluEval2'
    output_dir = './data/dataset/HaluEval2/'

    for file in os.listdir(data_dir):
        if file not in ['Education.json', 'Finance.json', 'Science.json']:
            continue

        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in tqdm(data):
            sentences = item['chatgpt_fact']
            sentence_labels = [e.lower() == 'true' for e in item['human_judge']]

            if len(sentences) == len(sentence_labels):
                sample = {
                    'question': item['user_query'],
                    'answer': item['chatgpt_response'],
                    'from': "HaluEval2-" + file.replace('.json', ''),
                    "sentences": sentences,
                    "sentence_labels": sentence_labels,
                    "label": all(sentence_labels)
                }
                samples.append(sample)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, file.replace(".json", ".jsonl")), 'w', encoding='utf-8') as fp:
            for sample in samples:
                fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
