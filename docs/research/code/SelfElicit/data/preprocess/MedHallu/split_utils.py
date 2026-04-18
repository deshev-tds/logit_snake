import re
import os
import json
from utils.llm import GPT4Model

gpt4_model = GPT4Model(None)


def check_keys(samples):
    required_keys = ['question', 'answer', 'claims', 'sentences', 'label', 'from', 'claim_labels', 'sentence_labels']
    for sample in samples:
        assert all([k in sample.keys() for k in required_keys])
        assert len(sample['claims']) == len(sample['claim_labels'])
        assert len(sample['sentences']) == len(sample['sentence_labels'])


def get_sentences(answer):
    sentences = re.split(r"。|\n", answer)  # split by '。' and '\n'
    sentences = list(filter(lambda s: len(s) >= 5, sentences))
    sentences = [s.strip() for s in sentences]
    return sentences


def write(samples, name, dir="./data/cache/MedHallu/export/"):
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, name), 'w', encoding='utf-8') as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")


def drop_short(sentences, min_length=5):
    return list(filter(lambda s: len(s) >= min_length, sentences))


ATOMIC_FACT_INSTRUCTION = """\
Instructions:
1. You are given a sentence. Your task is to break the sentence down into a \
list of atomic facts.
2. An atomic fact is a sentence containing a singular piece of information.
3. Each atomic fact in the outputted list should check a different piece of \
information.
4. Use the previous examples to learn how to do this.
5. You should only output the atomic facts as a list, with each item starting \
with "- ". Do not include other formatting.
6. Your task is to do this for the last sentence that is given.
"""
ATOMIC_FACT_PREFIX = """\
Please breakdown the following sentence into independent facts:
"""


def split_answer_with_gpt(answer: str, cache_filepath: str):
    with open("./data/preprocess/MedHallu/atomic_fact_few_shots.json", "r", encoding='utf-8') as fp:
        few_shots_samples = json.load(fp)

    while not os.path.exists(cache_filepath):
        prompt = ATOMIC_FACT_INSTRUCTION + "\n"
        for sample in few_shots_samples:
            prompt += ATOMIC_FACT_PREFIX + sample['answer'].replace("\n", "") + "\n"
            prompt += '\n'.join(["- " + c for c in sample['claims']])
            prompt += "\n\n"

        prompt += ATOMIC_FACT_PREFIX + answer.replace("\n", "")

        _, response, _ = gpt4_model.understand(prompt, history=[], targets=None)

        if len(response) > 0:
            sentences = response.splitlines()
            if len(sentences) > 0:
                with open(cache_filepath, 'w', encoding='utf-8') as fp:
                    fp.writelines(response)

    with open(cache_filepath, 'r', encoding='utf-8') as fp:
        sentences = fp.readlines()
        return sentences


ATOMIC_FACT_LABELING_INSTRUCTION = """\
Instructions:
1. You are given several sentences and a comment. \
The comment points out the incorrectness of some of the sentences. \
Your task is to find the incorrect sentence pointed out by the comment.
2. The sentences are given in a list. Each sentence starts with "- ".
3. The comment might include satisfaction/满意度 issues, correctness/专业性 issues, \
and universal/通用性 issues. But you should only focus on the correctness/专业性 issues.
4. Find the incorrect sentences pointed out by the comment. \
Note that in some cases all sentences might be correct and there is no incorrect sentence.
5. You should only copy the incorrect sentences as a list, with each item starting \
with "- ". Do not include other formatting. If there is no incorrect sentence, reply "- ". 
6. The sentences are annotated with <sentence>. The comment is annotated with <comment>. \
Your task is to do this for the given <sentence> and <comment>.
"""


def label_sentences_with_gpt(sentences: list[str], comment: str, cache_filepath: str):
    num_tries = 0
    while not os.path.exists(cache_filepath) and num_tries <= 20:
        print(cache_filepath)
        prompt = ATOMIC_FACT_LABELING_INSTRUCTION
        prompt += '<sentence>\n'
        for sentence in sentences:
            prompt += "- " + sentence + "\n"
        prompt += "<comment>\n" + comment + "\n"
        # prompt += "<incorrect sentence>\n"

        _, response, _ = gpt4_model.understand(prompt, history=[], targets=None)
        num_tries += 1

        if response:
            incorrect_sentences = response.splitlines()
            incorrect_sentences = [s.replace('- ', '') for s in incorrect_sentences]
            if any([s in orig for s in incorrect_sentences for orig in sentences]):
                with open(cache_filepath, 'w', encoding='utf-8') as fp:
                    fp.writelines(response)

    if num_tries > 5:
        raise ValueError(cache_filepath)

    with open(cache_filepath, 'r', encoding='utf-8') as fp:
        incorrect_sentences = fp.readlines()
        # if len(incorrect_sentences) > 6:
        #     raise ValueError(cache_filepath)

    def formatting(sentence: str):
        assert sentence.startswith('- ')
        return sentence[2:].replace('\n', '')

    incorrect_sentences = [formatting(s) for s in incorrect_sentences]
    labels = [True] * len(sentences)
    for s in incorrect_sentences:
        idx = list(filter(lambda i: s in sentences[i], range(len(sentences))))
        assert len(idx) > 0, cache_filepath
        for i in idx:
            labels[i] = False

    return labels
