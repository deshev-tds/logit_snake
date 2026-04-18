import json

from sympy.physics.units import quart

if __name__ == '__main__':

    with open("./data/original_dataset/WikiBio/dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open("./data/dataset/WikiBio/wiki_bio_gpt3_hallucination.jsonl", 'r', encoding='utf-8') as f:
        topics = [json.loads(line )['question'].replace('This is a Wikipedia passage about ','').replace(':', '') for line in f.readlines()]

    samples = []
    for sample, topic in zip(data, topics):
        sentence_labels = []
        for annotate in sample['annotation']:
            if annotate == 'major_inaccurate':
                sentence_labels.append(0)
            elif annotate == 'minor_inaccurate':
                sentence_labels.append(0.5)
            else:
                sentence_labels.append(1)

        print(topic)
        question = f"Please give me an introduction about {topic}"
        answer = sample['gpt3_text']
        sentences = [s.replace('He', topic).replace('She', topic) for s in sample['gpt3_sentences']]

        sample = {
            'question': question,
            'answer': answer,
            "sentences": sentences,
            "sentence_labels": sentence_labels,
            "label": all(sentence_labels),
            "from": "WikiBio"
        }
        samples.append(sample)

    with open("./data/dataset/WikiBio/wiki_bio_gpt3_hallucination2.jsonl", 'w', encoding='utf-8') as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
