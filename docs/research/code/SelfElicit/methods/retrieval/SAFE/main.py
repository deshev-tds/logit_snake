import numpy as np

from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from util import extract_first_square_brackets, extract_first_code_block
from revise_template import *
from relevance_template import *
from retrieval_template import *


class SAFE(AbstractMethod):
    """
    Long-form factuality in large language models
    """

    def revise_fact(self, model, response, atomic_fact):
        """Modify the atomic fact to be self-contained."""
        # speedup
        if all([p not in atomic_fact for p in PRONOUNS]):
            return atomic_fact

        few_shots = []
        for case in REVISE_FEW_SHOTS:
            few_shots.append("\n".join([case['header'], case['answer'], case['statement'], case['output']]))

        full_prompt = REVISE_TEMPLATE
        full_prompt = full_prompt.replace('##STATEMENT##', atomic_fact)
        full_prompt = full_prompt.replace("##RESPONSE##", response)
        full_prompt = full_prompt.replace("##FEW_SHOTS##", "\n\n".join(few_shots))
        full_prompt = full_prompt.strip('\n')

        _, response, _ = model.understand(full_prompt, history=[], targets=None)
        revised_fact = extract_first_code_block(response, ignore_language=True)

        return revised_fact or atomic_fact

    def check_relevance(self, model, question: str, response: str, atomic_fact: str):
        """Check if the atomic fact is relevant for answering the prompt."""
        few_shots = []
        for case in RELEVANCE_FEW_SHOTS:
            few_shots.append("\n".join([case['header'], case['question'], case['response'], case['statement'], case['output']]))

        full_prompt = RELEVANCE_TEMPLATE
        full_prompt = full_prompt.replace("##QUESTION##", question)
        full_prompt = full_prompt.replace("##RESPONSE##", response)
        full_prompt = full_prompt.replace("##STATEMENT##", atomic_fact)
        full_prompt = full_prompt.replace("##FEW_SHOTS##", "\n\n".join(few_shots))
        full_prompt = full_prompt.strip('\n')

        def post_process(prob, response, output):
            answer = extract_first_square_brackets(response)
            assert answer in RELEVANCE_TARGETS
            return prob, answer, output

        _, answer, _ = model.understand(full_prompt, post_process=post_process)
        return answer == RELEVANCE_TARGETS[0]

    def check_atomic_fact(self, model, atomic_fact: str, knowledge: str):
        """Check if the given atomic fact is supported."""
        full_prompt = FINAL_ANSWER_TEMPLATE
        full_prompt = full_prompt.replace("##KNOWLEDGE##", knowledge)
        full_prompt = full_prompt.replace("##STATEMENT##", atomic_fact)
        full_prompt = full_prompt.strip('\n')
        _, response, _ = model.understand(full_prompt)
        answer = extract_first_square_brackets(response)

        if answer in FINAL_ANSWER_TARGETS:
            return answer != FINAL_ANSWER_TARGETS[1]
        else:
            return True  # default True

    def __call__(self, sample, model, **kwargs):
        question = sample['question'].replace("\n", "")
        answer = sample['answer'].replace("\n", "")

        probs, responses = [], []
        for sentence, references in zip(sample['sentences'], sample['references']):
            revised_sentence = self.revise_fact(model, answer, sentence)

            is_relevant = self.check_relevance(model, question, answer, revised_sentence)
            if not is_relevant:
                probs.append(inverse_softmax(0))  # regard as True
                responses.append(f"Not relevant")
            else:
                scores = []
                for reference in references[:self.args.n_retrieval]:
                    is_supported = self.check_atomic_fact(model, revised_sentence, reference)
                    scores.append(is_supported)
                score_agg = all(scores)

                probs.append(inverse_softmax(1 - score_agg))
                responses.append(f"Relevant. With {scores}, aggregated score {score_agg}")

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--n_retrieval", type=int, default=5)

    args = parse_args(parser)
    init_env(args, "safe")

    method = SAFE(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
