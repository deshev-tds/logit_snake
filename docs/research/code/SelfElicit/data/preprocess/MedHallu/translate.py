import os
import re
import json
import argparse
from tqdm import tqdm
from split_utils import write
from utils.llm import QWen2Model
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--server", type=str, help="API server")

args = parser.parse_args()
args.model_path = args.model_path.rstrip('/')
model = QWen2Model(args)
model.verbose = True
model.set_generation_config('do_sample', False)
model.set_generation_config('max_new_tokens', 8192)


def translate(sentences):
    if isinstance(sentences, str):
        prompt = "Please translate the following sentence about medical into English.\n"
        prompt += "Only output the translated sentence and nothing else."
        prompt += "Sentences to be translated into English:\n"
        prompt += sentences
        _, response, _ = model.understand(prompt)

    elif isinstance(sentences, list):
        prompt = "Please translate the following sentences about medical into English.\n"
        prompt += "Only output the translated sentences with serial number and nothing else."
        prompt += "Sentences to be translated into English:\n"
        for i, sentence in enumerate(sentences):
            sentence = re.sub(r"^\d+\.", "", sentence).strip()
            prompt += f"{i + 1}. {sentence}\n"
        prompt += "Translate:"

        def post_process(prob, response, output):
            response = re.findall(r"\d+\.\s+(.*)", response)
            assert len(response) == len(sentences)
            return prob, response, output

        _, response, _ = model.understand(prompt, post_process=post_process)
    
    return response


if __name__ == '__main__':
    path = "./data/cache/MedHallu/export"
    files = list(filter(lambda file: os.path.splitext(file)[-1] == '.jsonl', os.listdir(path)))
    files = sorted(files)

    all_samples = {}
    for file in files:
        print(f"Loading datasets from {file}")

        samples = []
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                sample = json.loads(line)
                assert all([k in sample for k in
                            ['question', 'answer', 'claims', 'sentences', 'label', 'from', 'claim_labels',
                             'sentence_labels']])
                samples.append(sample)

        file = file.replace('.jsonl', '')
        all_samples[file] = samples

    for file in all_samples:
        cache_path = "./data/cache/MedHallu/translate/"
        os.makedirs(os.path.join(cache_path, file), exist_ok=True)

        new_samples = []
        for i, sample in enumerate(tqdm(all_samples[file], desc=file)):
            cache_file = os.path.join(cache_path, file, f"{i}.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding='utf-8') as f:
                    sample = json.load(f)

            else:
                # todo may split the claims in English
                sample['question'] = translate(sample['question'])
                sample['answer'] = translate(sample['answer'])
                sample['claims'] = translate(sample['claims'])
                sample['sentences'] = translate(sample['sentences'])

                # claim_retrievals = [online_kuake_search(c) for c in sample['claims']]
                # sentence_retrievals = [online_kuake_search(s) for s in sample['sentences']]
                # claim_retrievals = [[ele['snippet'].replace('\n', '') for ele in item] for item in claim_retrievals]
                # sentence_retrievals = [[ele['snippet'].replace('\n', '') for ele in item] for item in sentence_retrievals]
                # if len(claim_retrievals) == len(sample['claims']) and len(sentence_retrievals) == len(sample['sentences']):
                #     print(cache_file)
                #     os.remove(cache_file)
                # else:
                #     sample['claim_retrievals'] = claim_retrievals
                #     sample['sentence_retrievals'] = sentence_retrievals

                with open(cache_file, "w", encoding='utf-8') as f:
                    json.dump(sample, f, ensure_ascii=False)

            new_samples.append(sample)

        write(new_samples, file + ".jsonl", dir="./data/dataset/MedHallu_en/")
