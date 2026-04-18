import os
import json
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template import *
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from utils.llm import GPT4Model
from methods.retrieval.FActScore.main import FActScore
from utils.wrapper import ModelConfiguration


class FactTune(AbstractMethod):
    """
    Fine-tuning Language Models for Factuality
    """

    def __init__(self, args):
        super().__init__(args)

        self.fact_score_model = FActScore(args)

        self.question_generation_model = GPT4Model(args)

    def support_check(self, query, references, model):
        is_supported, response = self.fact_score_model.get_score(query, references, model)

        return is_supported

    def question_generation(self, queries, sample_id):
        cache_folder = os.path.join("./cache/FLMF/question_generation")
        os.makedirs(cache_folder, exist_ok=True)
        cache_file = os.path.join(cache_folder, f"{sample_id}.json")

        few_shot = ""
        for item in QUESTION_GENERATION_FEW_SHOTS[:self.args.n_question_gen_few_shots]:
            shot = [item['instruction']]
            for case in item['few_shots']:
                shot.append(case['claim'])
                shot.append(case['question'])
            shot = "\n".join(shot)
            few_shot += shot + "\n"

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding='utf-8') as fp:
                questions = json.load(fp)

        else:
            questions = []
            for query in queries:
                prompt = QUESTION_GENERATION_TEMPLATE.replace("##FEW_SHOTS##", few_shot)
                prompt = prompt.replace("##QUERY##", query)

                _, question, _ = self.question_generation_model.understand(query=prompt,
                                                                           history=[],
                                                                           targets=None)

                questions.append(question)

            with open(cache_file, "w", encoding='utf-8') as fp:
                json.dump(questions, fp, ensure_ascii=False)

        questions = [q.replace("问题：", "") for q in questions]
        return questions

    def answer_question(self, question, model):
        responses = []
        with ModelConfiguration(model, config_name='do_sample', value=True):
            for i_try in range(self.args.n_answers):
                _, response, _ = model.understand(query=question, history=[], targets=None)
                responses.append(response.replace("\n", ""))

        return responses

    def compare_answers(self, query, answers, model):
        scores = []
        for answer in answers:
            few_shots = ""
            for case in EQUIVALENCE_CHECK_FEW_SHOTS[:self.args.n_equivalence_few_shots]:
                few_shots += "\n".join([case['query_a'], case['query_b'], case['output']])
                few_shots += "\n\n"

            prompt = EQUIVALENCE_CHECK_TEMPLATE.replace("##FEW_SHOTS##", few_shots)
            prompt = prompt.replace("##QUERY_A##", query)
            prompt = prompt.replace("##QUERY_B##", answer)

            _, response, _ = model.understand(query=prompt, history=[], targets=None)

            if response.startswith(EQUIVALENCE_CHECK_TARGETS[0]):
                scores.append(True)
            elif response.startswith(EQUIVALENCE_CHECK_TARGETS[1]):
                scores.append(False)
            elif EQUIVALENCE_CHECK_TARGETS[1] in response:
                scores.append(False)
            else:
                scores.append(True)

        scores = np.mean(scores).item()
        return scores

    def __call__(self, sample, model, **kwargs):
        sample_id = kwargs['sample_id']

        questions = self.question_generation(sample['sentences'], sample_id)

        probs, responses = [], []
        for cid in range(len(sample['sentences'])):
            # reference-based
            is_supported = self.support_check(query=sample['sentences'][cid],
                                              references=sample['references'][cid],
                                              model=model)

            if is_supported:
                probs.append(inverse_softmax(0))
                responses.append("This claim is supported by the reference")
            else:
                # reference-free
                answers = self.answer_question(question=questions[cid],
                                               model=model)
                score = self.compare_answers(query=sample['sentences'][cid],
                                             answers=answers,
                                             model=model)
                probs.append(inverse_softmax(1 - score))
                responses.append(f"The support score over {len(answers)} answers is {1 - score}")

        return np.stack(probs), responses


if __name__ == '__main__':
    # factscore arguments
    parser.add_argument("--n_retrieval", type=int, default=5)
    parser.add_argument("--npm_model_path", type=str)
    parser.add_argument("--npm_thres", type=float, default=0.005)

    parser.add_argument("--n_answers", type=int, default=20)
    parser.add_argument("--n_question_gen_few_shots", type=int, default=2)
    parser.add_argument("--n_equivalence_few_shots", type=int, default=4)

    args = parse_args(parser)
    init_env(args, "fact_tune")

    method = FactTune(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
