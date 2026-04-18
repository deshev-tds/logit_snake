import os
import re
import json
import argparse
from tqdm import tqdm
from data.dataloader import load_dataset
from utils.llm import load_model
from methods.thought.extract import _extract_knowledge


def get_core_sentences(sentences, pred_clses):
    RETAINED_CLASSES = [0, 5]
    core_sentences = []
    for sentence, cls in zip(sentences, pred_clses):
        if cls in RETAINED_CLASSES:
            core_sentences.append(sentence)
        else:
            core_sentences.append(None)
    return core_sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'val'])

    parser.add_argument("--data_path", type=str, default="./data/dataset/MedHallu/")
    parser.add_argument("--cls_path", type=str, default="./cache/classify/qwen2_72b_chat/")
    parser.add_argument("--extc_path", type=str, default="./cache/extract/")

    args = parser.parse_args()
    args.do_sample = False
    args.max_new_tokens = 8192
    args.top_k = args.top_p = args.repetition_penalty = None
    args.model_path = args.model_path.rstrip('/')
    args.model_name = args.model_path.split('/')[-1].lower().replace('-', '_')

    model = load_model(args.__dict__)
    datasets = load_dataset(args.data_path)

    cls_path = os.path.join(args.cls_path, args.mode)
    extc_path = os.path.join(args.extc_path, args.model_name, args.mode)
    os.makedirs(extc_path, exist_ok=True)

    for i in tqdm(range(len(datasets[args.mode]))):
        sample = datasets[args.mode][i]
        cls_cache_file = os.path.join(cls_path, f"{i}.json")
        extc_cache_file = os.path.join(extc_path, f"{i}.json")

        if not os.path.exists(extc_cache_file):
            with open(cls_cache_file, "r", encoding='utf-8') as fp:
                cache = json.load(fp)
                sentences = cache['sentences']
                clses = cache['clses']

            assert [_ == __ for _, __ in zip(sentences, sample['sentences'])]
            core_sentences = get_core_sentences(sentences, clses)
            core_sentences, knowledges, entities, cost = _extract_knowledge(core_sentences, 'global', model)

            with open(extc_cache_file, "w", encoding='utf-8') as fp:
                json.dump({'sentences': sentences,
                           'core_sentences': core_sentences, 
                           'entities': entities,
                           'knowledges': knowledges, 
                           'cost': cost}, fp, ensure_ascii=False)
