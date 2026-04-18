import json
import os
import random
from collections import defaultdict
import numpy as np


def load_raw_dataset(path: str, verbose: bool = False):
    files = list(filter(lambda file: os.path.splitext(file)[-1] == '.jsonl', os.listdir(path)))
    files = sorted(files)

    all_samples = []
    from_count = defaultdict(int)
    for file in files:
        print(f"Loading datasets from {file}")

        has_claim = False
        samples = []
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                sample = json.loads(line)
                assert all([k in sample for k in ['question', 'answer', 'sentences', 'label', 'sentence_labels']])
                has_claim = 'claims' in sample and 'claim_labels' in sample
                samples.append(sample)
                from_count[sample['from']] += 1

        if verbose:
            print(f"Loaded {len(samples)} samples from {file}")
            print(f"Query label: {np.unique([s['label'] for s in samples], return_counts=True)}")
            print(f"Sentence count: {np.sum([len(s['sentences']) for s in samples])}")
            print(f"Sentence label: {np.unique([l for s in samples for l in s['sentence_labels']], return_counts=True)}")
            if has_claim:
                print(f"Claim count: {np.sum([len(s['claims']) for s in samples])}")
                print(f"Claim label: {np.unique([l for s in samples for l in s['claim_labels']], return_counts=True)}")
            print("\n")
        all_samples.extend(samples)

    if verbose:
        print(from_count)
    return all_samples


def split_dataset(samples: list, split: list):
    corpus = samples

    random.seed(0)
    random.shuffle(corpus)
    train_size = int(split[0] * len(corpus))
    val_size = int(split[1] * len(corpus))
    train_set = corpus[:train_size]
    val_set = corpus[train_size:train_size + val_size]
    test_set = corpus[train_size + val_size:]

    return {'train': train_set,
            'val': val_set,
            'test': test_set}


def drop_duplicate(samples: list):
    get_index = lambda item: f"{item['question']}-{item['answer']}".replace('\n', '')
    drop_count = 0
    table = set()

    nodup_samples = []
    for sample in samples:
        index = get_index(sample)
        if index not in table:
            nodup_samples.append(sample)
            table.add(index)
        else:
            # print(f"{index} is duplicated, dropped")
            drop_count += 1

    print(f"{drop_count} samples are duplicated")
    return nodup_samples


def load_dataset(path, verbose=False):
    samples = load_raw_dataset(path, verbose=verbose)
    samples = drop_duplicate(samples)

    split = {
        "MedHallu": [0.6, 0.1, 0.3],
        "MedHallu_en": [0.6, 0.1, 0.3],
        "FActScore": [0, 0.3, 0.7],
        "WikiBio": [0, 0.3, 0.7],
        "HaluEval2": [0, 0.3, 0.7]
    }
    datasets = split_dataset(samples, split=split[path.rstrip("/").split('/')[-1]])

    return datasets


def print_features(samples):
    has_claim = 'claim' in samples[0] and 'claim_labels' in samples[0]
    print("\n\n")
    print(len(samples))
    print(f"Query label: {np.unique([s['label'] for s in samples], return_counts=True)}")
    print(f"Sentence count: {np.sum([len(s['sentences']) for s in samples])}")
    print(f"Sentence label: {np.unique([l for s in samples for l in s['sentence_labels']], return_counts=True)}")
    if has_claim:
        print(f"Claim count: {np.sum([len(s['claims']) for s in samples])}")
        print(f"Claim label: {np.unique([l for s in samples for l in s['claim_labels']], return_counts=True)}")
    print(random.choice(samples))


if __name__ == '__main__':
    datasets = load_dataset("./data/dataset/AliHealthQA/")
    test_set = datasets['test']
    print_features(test_set)
