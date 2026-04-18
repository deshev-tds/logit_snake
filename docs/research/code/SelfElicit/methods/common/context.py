import re
import os
import json
from .context_template_zh import PRONOUN_REPLACEMENT_TEMPLATE_ZH
from .context_template_en import PRONOUN_REPLACEMENT_TEMPLATE_EN
from utils.utils import match


def has_pronoun(sentence, language):
    if language == 'zh':
        pronouns = ['它们', '以上', '这些', '这种', '那些', '这类', '他们']
        patterns = [r"[^尤]其[^他它中余间次]", r"[^其]它", r"^它", r"^这"]
    else:
        pronouns = ['they', 'above', 'these', 'this', 'those', 'them', 'others', 'its', 'their',
                    'both', 'either', 'some']
        patterns = [r"^it [^i]"]

    for word in pronouns:
        if match(word, sentence, language):
            return True

    for pattern in patterns:
        if re.match(pattern, sentence, re.IGNORECASE):
            return True

    return False


def remove_pronoun(sentence, context, model, language):
    if language == 'zh':
        prompt = PRONOUN_REPLACEMENT_TEMPLATE_ZH
        prompt = prompt.replace("##CONTEXT##", "上下文：" + "。".join(context))
        prompt = prompt.replace("##SENTENCE##", "句子：" + sentence)
        pattern = r'替换后的句子是?：?(.*)'
    else:
        prompt = PRONOUN_REPLACEMENT_TEMPLATE_EN
        prompt = prompt.replace("##CONTEXT##", "Context: " + ".".join(context))
        prompt = prompt.replace("##SENTENCE##", "Sentence: " + sentence)
        pattern = r'The updated sentence.*: ?(.*)'

    def post_process(prob, response, output):
        match = re.findall(pattern, response.replace("\n", ""))
        assert match
        match = match[-1]
        match = re.sub(r"[“”\"\"\'\']", "", match)
        return prob, match, output

    try:
        _, response, output = model.understand(prompt, post_process=post_process)
    except:  # if no need for pronoun removal
        response, output = sentence, None

    return response, output


def update_context(sentences, is_core_sentences, model, sample_id, context_cache=None, language='zh'):
    cache_file = os.path.join(str(context_cache), f"{sample_id}.json")

    if context_cache:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding='utf-8') as fp:
                cache = json.load(fp)
                updated_sentences = cache["updated_sentences"]
                for cost in cache['cost']:
                    model.counter.step(cost)

        else:
            cost = []
            updated_sentences = []
            for i in range(len(sentences)):
                if is_core_sentences[i] and has_pronoun(sentences[i], language):
                    updated_sentence, output = remove_pronoun(sentences[i], [s for s in sentences if s], model, language)
                    updated_sentences.append(updated_sentence)
                    if output:
                        cost.append(len(output.scores))
                else:
                    updated_sentences.append(sentences[i])

            os.makedirs(context_cache, exist_ok=True)
            with open(cache_file, "w", encoding='utf-8') as fp:
                json.dump({'orig_sentences': sentences, 'updated_sentences': updated_sentences, 'cost': cost},
                          fp, ensure_ascii=False)

        assert len(sentences) == len(updated_sentences) == len(is_core_sentences)
        for i in range(len(sentences)):
            if updated_sentences[i] and is_core_sentences[i]:
                sentences[i] = updated_sentences[i]

    return sentences
