import torch
import numpy as np
from .extract import *
from .judge_template_zh import JUDGE_SENTENCE_TEMPLATE_ZH, JUDGE_TARGETS_ZH
from .judge_template_en import JUDGE_SENTENCE_TEMPLATE_EN, JUDGE_TARGETS_EN
from utils.utils import inverse_softmax


def judge_sentence(sentence, model, language, **kwargs):
    targets = JUDGE_TARGETS_ZH if language == 'zh' else JUDGE_TARGETS_EN
    if 'context' in kwargs and isinstance(kwargs['context'], list) and len(kwargs['context']) > 0:
        if language == 'zh':
            prompt = JUDGE_SENTENCE_TEMPLATE_ZH.replace("##CONTEXT##", "上下文：" + "。".join(kwargs['context']) + "\n")
        else:
            prompt = JUDGE_SENTENCE_TEMPLATE_EN.replace("##CONTEXT##", "Context: " + ". ".join(kwargs['context']) + "\n")
    else:
        if language == 'zh':
            prompt = JUDGE_SENTENCE_TEMPLATE_ZH.replace("##CONTEXT##", "")
        else:
            prompt = JUDGE_SENTENCE_TEMPLATE_EN.replace("##CONTEXT##", "")

    prompt = prompt.replace("##QUERY##", sentence)
    prompt = prompt.replace("##TARGETS##", ' '.join(targets))

    prob, response, _ = model.understand(query=prompt, targets=targets)
    if torch.isinf(prob).all():
        prob = torch.tensor([1, -torch.inf, -torch.inf])
    prob = prob.float().cpu().numpy()
    # response = f"Judge as {JUDGE_TARGETS[prob.argmax(axis=0).item()]}"

    return prob, response


def judge_knowledge(knowledges, model, language, **kwargs):
    if knowledges:
        prob_knowledges = []
        for knowledge in knowledges:
            prob, _ = judge_sentence(knowledge, model, language, **kwargs)
            prob_knowledges.append(torch.from_numpy(prob))

        prob_knowledges = torch.softmax(torch.stack(prob_knowledges, dim=0), dim=-1)
        prob = prob_knowledges[:, 1].cpu().numpy()
        prob = np.max(prob).item()
        response = f"Aggregate {knowledges} into {prob}"

        prob = inverse_softmax(prob)

    else:
        prob = inverse_softmax(0.)
        response = f"No knowledges"

    return prob, response


def judge_knowledge_question(knowledges, model, language, **kwargs):
    raise NotImplementedError()
    assert language == 'zh'
    if knowledges:
        prob_knowledges = []
        for knowledge in knowledges:
            prompt = QUESTION_GENERATION_TEMPLATE.replace("##SENTENCE##", knowledge)
            _, question, _ = model.understand(query=prompt)

            _, answer, _ = model.understand(query=question)

            compare = COMPARE_TEMPLATE.replace("##REFERENCES##", answer.replace("\n", "")).replace("##SENTENCE##", knowledge)
            _, response, _ = model.understand(query=compare)

            if '支持' in response and '不支持' not in response:
                prob = 0
            elif '矛盾' in response and '不矛盾' not in response:
                prob = 1
            else:
                prob = 0

            prob_knowledges.append(prob)

        prob = np.max(prob_knowledges).item()
    else:
        prob = 0.

    prob = inverse_softmax(prob)
    response = f"Agg {knowledges} into {prob}"

    return prob, response
