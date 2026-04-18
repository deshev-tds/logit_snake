import os
import re
import json
from .classify_template_zh import CLASSIFICATION_TEMPLATE_ZH, CLASSIFICATION_TARGETS_ZH, RETAINED_CLASSES, DEFAULT_CLASS
from .classify_template_en import CLASSIFICATION_TEMPLATE_EN, CLASSIFICATION_TARGETS_EN


def _classify_sentence(sentence, model, language):
    prompt = CLASSIFICATION_TEMPLATE_ZH if language == 'zh' else CLASSIFICATION_TEMPLATE_EN
    prompt = prompt.replace("##SENTENCE##", sentence)
    targets = CLASSIFICATION_TARGETS_ZH if language == 'zh' else CLASSIFICATION_TARGETS_EN
    _, response, outputs = model.understand(query=prompt, targets=targets)

    pred_cls = DEFAULT_CLASS
    for i, target in enumerate(targets):
        if target in response:
            pred_cls = i
            break

    response = response.replace("\n", "") + f"Classify as {targets[pred_cls]}"
    return pred_cls, response, outputs


def classify_rules(sentence, language, verbose=False):
    if language == 'zh':
        rule2_patterns = ["几种", '以下', '如下', '下列', '以上', '参考自', "综合考虑", "综合评估"]
        rule3_patterns = ['医生', '专家', '医师', '家人', '医嘱', '心情', '就医', '心态', '个性化', '个体']
    else:
        rule2_patterns = ["several", 'following', 'above', "comprehensive", "considerate"]
        rule3_patterns = ['doctor', 'physician', 'surgeon', 'family', 'friend', 'mood', 'hospital', 'attitude', 'personal', 'individual']

    if any(w in sentence.lower() for w in rule2_patterns):
        pred_cls = 2
        response = f"Key word classify as 综述概述型/Summarization Type"

    elif any(w in sentence.lower() for w in rule3_patterns):
        pred_cls = 3
        response = f"Key word classify as 就医型/Hospitalization Type"

    else:
        return None

    if verbose:
        print(response)

    return (pred_cls, response)


def classify_sentence_with_rules(sentences, model, language):
    cost = []
    pred_clses, responses = [], []
    for sentence in sentences:
        rule_matching_result = classify_rules(sentence, language, verbose=model.verbose)
        if rule_matching_result:
            pred_cls, response = rule_matching_result
        else:
            pred_cls, response, outputs = _classify_sentence(sentence, model, language)
            cost.append(len(outputs.scores))

        pred_clses.append(pred_cls)
        responses.append(response)

    return pred_clses, responses, cost


def classify_sentences(sentences, model, sample_id,
                       cls_cache=None, language='zh'):
    cache_file = os.path.join(str(cls_cache), f"{sample_id}.json")

    if cls_cache:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding='utf-8') as fp:
                cache = json.load(fp)
                pred_clses = cache['pred_cls']
                responses = cache['responses']
                for cost in cache['cost']:
                    model.counter.step(cost)

        else:
            pred_clses, responses, cost = classify_sentence_with_rules(sentences, model, language)

            os.makedirs(cls_cache, exist_ok=True)
            with open(cache_file, "w", encoding='utf-8') as fp:
                json.dump({'pred_cls': pred_clses, 'responses': responses, 'cost': cost},
                          fp, ensure_ascii=False)

        is_core_sentences = []
        for sentence, cls in zip(sentences, pred_clses):
            if cls in RETAINED_CLASSES:
                is_core_sentences.append(True)
            else:
                is_core_sentences.append(False)

    else:
        is_core_sentences = [True] * len(sentences)

    return is_core_sentences
