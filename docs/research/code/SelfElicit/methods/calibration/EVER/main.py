import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import *
from template_en import *
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result


class EVER(AbstractMethod):
    """
    Ever: Mitigating Hallucination in Large Language Models through Real-Time Verification and Rectification
    """

    def __init__(self, args):
        super().__init__(args)

        self.extract_concept_template = EXTRACT_CONCEPTS_TEMPLATE_ZH if args.language == 'zh' else EXTRACT_CONCEPTS_TEMPLATE_EN
        self.extract_concept_fewshots = EXTRACT_CONCEPTS_FEW_SHOTS_ZH if args.language == 'zh' else EXTRACT_CONCEPTS_FEW_SHOTS_EN
        self.extract_concept_fewshots = self.extract_concept_fewshots[:self.args.n_extract_concept_few_shots]

        self.question_generation_template = QUESTION_GENERATION_TEMPLATE_ZH if args.language == 'zh' else QUESTION_GENERATION_TEMPLATE_EN
        self.question_generation_fewshots = QUESTION_GENERATION_FEW_SHOTS_ZH if args.language == 'zh' else QUESTION_GENERATION_FEW_SHOTS_EN
        self.question_generation_fewshots = self.question_generation_fewshots[:self.args.n_question_generation_few_shots]

        if self.args.checking_with_retrieval:
            self.question_checking_template = QUESTION_CHECKING_TEMPLATE_RETRIEVAL_ZH if args.language == 'zh' else QUESTION_CHECKING_TEMPLATE_RETRIEVAL_EN
        else:
            self.question_checking_template = QUESTION_CHECKING_TEMPLATE_ZH if args.language == 'zh' else QUESTION_CHECKING_TEMPLATE_EN
        self.question_checking_fewshots = QUESTION_CHECKING_FEW_SHOTS_ZH if args.language == 'zh' else QUESTION_CHECKING_FEW_SHOTS_EN
        self.question_checking_fewshots = self.question_checking_fewshots[:self.args.n_question_checking_few_shots]
        self.question_checking_targets = QUESTION_CHECKING_TARGETS_ZH if args.language == 'zh' else QUESTION_CHECKING_TARGETS_EN

        self.checking_with_retrieval = args.checking_with_retrieval

    def identify_entities(self, query, model):
        few_shots = []
        for item in self.extract_concept_fewshots:
            if self.args.language == 'zh':
                shot = f"例子{len(few_shots) + 1}：\n"
                shot += f"句子：{item['query']}\n"
                shot += f"概念：{item['concept']}\n"
            else:
                shot = f"Example {len(few_shots) + 1}:\n"
                shot += f"Sentence: {item['query']}\n"
                shot += f"Concepts: {item['concept']}\n"
            few_shots.append(shot)

        prompt = self.extract_concept_template
        prompt = prompt.replace("##QUERY##", query)
        prompt = prompt.replace("##FEWSHOTS##", "\n".join(few_shots))
        _, response, _ = model.understand(query=prompt)

        if args.language == 'zh':
            entities = re.split(r"[;；]", response)
            entities = list(filter(lambda ent: len(ent) < 10, entities))
        else:
            entities = re.findall(r"\*\s*(.*)", response)
            entities = list(filter(lambda ent: len(ent.split()) < 5, entities))
        entities = list(filter(lambda ent: not any([ent.startswith(w) for w in ['无', '没有', 'No']]), entities))
        return entities

    def question_generation(self, query, concept, model):
        few_shots = []
        for item in self.question_generation_fewshots:
            shot = self.question_generation_template
            shot = shot.replace("##QUERY##", item['query'])
            shot = shot.replace("##CONCEPT##", item['concept'])
            few_shots.append((shot, item['question']))

        prompt = self.question_generation_template
        prompt = prompt.replace("##QUERY##", query)
        prompt = prompt.replace("##CONCEPT##", concept)
        _, question, _ = model.understand(query=prompt, history=few_shots)

        return question

    def support_checking(self, question, retrievals, model):
        if self.checking_with_retrieval:
            for retrieval in retrievals:
                few_shots = []
                for item in self.question_checking_fewshots[:self.args.n_question_checking_few_shots]:
                    shot = self.question_checking_template
                    shot = shot.replace("##REFERENCE##", item['reference'])
                    shot = shot.replace("##QUESTION##", item['question'])
                    few_shots.append((shot, item['answer_retrieval']))

                prompt = self.question_checking_template
                prompt = prompt.replace("##REFERENCE##", retrieval)
                prompt = prompt.replace("##QUESTION##", question)

                _, response, _ = model.understand(query=prompt,
                                                  history=few_shots,
                                                  targets=self.question_checking_targets)

                if self.question_checking_targets[1] in response and self.question_checking_targets[0] not in response:
                    return False

        else:
            few_shots = []
            for item in self.question_checking_fewshots:
                shot = self.question_checking_template
                shot = shot.replace("##QUESTION##", item['question'])
                few_shots.append((shot, item['answer']))

            prompt = self.question_checking_template
            prompt = prompt.replace("##QUESTION##", question)
            _, response, _ = model.understand(query=prompt,
                                              history=few_shots,
                                              targets=self.question_checking_targets)

            if self.question_checking_targets[1] in response and self.question_checking_targets[0] not in response:
                return False

        return True

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        for sid, (sentence, is_core_sentence) in enumerate(zip(sentences, is_core_sentences)):
            if is_core_sentence:
                entities = self.identify_entities(sentence, model)
                retrievals = sample['references'][sid] if sample['references'] else []

                supports = []
                for entity in entities:
                    question = self.question_generation(sentence, entity, model)

                    support = self.support_checking(question, retrievals, model)
                    supports.append(support)

                if supports:
                    prob = 1 - np.min(supports)
                else:
                    prob = 0
                
                prob = inverse_softmax(prob)
                response = f"Find {len(entities)} entities {entities} with prediction {prob}."

            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--n_extract_concept_few_shots", type=int, default=5)
    parser.add_argument("--n_question_generation_few_shots", type=int, default=4)
    parser.add_argument("--n_question_checking_few_shots", type=int, default=3)
    parser.add_argument("--checking_with_retrieval", action="store_true")

    args = parse_args(parser)
    init_env(args, "EVER", postfix='retrieval' if args.checking_with_retrieval else None)

    method = EVER(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
