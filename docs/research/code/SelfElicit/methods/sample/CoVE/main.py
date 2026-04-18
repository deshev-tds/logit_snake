import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import *
from template_en import *
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result


class CoVE(AbstractMethod):
    """
    Chain-of-Verification Reduces Hallucination in Large Language Models
    """

    def __init__(self, args):
        super().__init__(args)

        self.execute_method = args.execute_method

        self.plan_template = PLAN_TEMPLATE_ZH if args.language == 'zh' else PLAN_TEMPLATE_EN

        self.execute_template_2step = EXECUTE_TEMPLATE_2STEP_ZH if args.language == 'zh' else EXECUTE_TEMPLATE_2STEP_EN
        self.execute_template_factored = EXECUTE_TEMPLATE_FACTOR_ZH if args.language == 'zh' else EXECUTE_TEMPLATE_FACTOR_EN

        self.verify_template = VERIFY_TEMPLATE_ZH if args.language == 'zh' else VERIFY_TEMPLATE_EN
        self.verify_block = VERIFY_BLOCK_ZH if args.language == 'zh' else VERIFY_BLOCK_EN
        self.verify_targets = VERIFY_TARGETS_ZH if args.language == 'zh' else VERIFY_TARGETS_EN

    def execute(self, questions, model):
        if self.execute_method == '2step':
            block = [f"{i + 1}. " + q.replace("\n", "") for i, q in enumerate(questions)]
            prompt = self.execute_template_2step
            prompt = prompt.replace("##QUESTIONS", "\n".join(block))

            def post_execute(prob, response, output):
                answers = re.split(r"\d+\. ", response)[1:]
                assert len(answers) == len(questions)
                return prob, answers, output

            _, answers, _ = model.understand(prompt, post_process=post_execute)
            return [(q, a) for q, a in zip(questions, answers)]

        elif self.execute_method == 'factored':
            qa = []
            for question in questions:
                prompt = self.execute_template_factored
                prompt = prompt.replace("##QUESTION##", question)
                _, answer, _ = model.understand(prompt)
                qa.append((question, answer))

        else:
            raise NotImplementedError

        return qa

    def cove_single(self, sentence, model):
        # plan
        prompt = self.plan_template
        prompt = prompt.replace("##SENTENCE##", sentence)
        _, response, _ = model.understand(prompt)
        questions = re.findall(r"\*\s+(.*)", response)
        if len(questions) == 0:
            questions = re.findall(r"\d+\.\s+(.*)", response)

        # execute
        if len(questions) == 0:
            return get_default_result()

        qa = self.execute(questions, model)

        # verify
        def post_verify(prob, response, output):
            assert any(k in response.lower() for k in self.verify_targets[0] + self.verify_targets[1])
            return prob, response, output

        prompt = self.verify_template
        prompt = prompt.replace("##SENTENCE##", sentence)
        qa_blocks = [self.verify_block.replace("##QUESTION##", q).replace("##ANSWER##", a) for (q, a) in qa]
        prompt = prompt.replace("##BLOCKS##", "\n".join(qa_blocks))
        _, response, _ = model.understand(prompt, post_process=post_verify)

        if any(k in response.lower() for k in self.verify_targets[1]):
            prob = 1
        elif any(k in response.lower() for k in self.verify_targets[0]):
            prob = 0
        else:
            prob = 0.5

        return inverse_softmax(prob), response

    def __call__(self, sample, model, **kwargs):
        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        probs, responses = [], []
        for sentence, is_core_sentence in zip(sentences, is_core_sentences):
            if is_core_sentence:
                prob, response = self.cove_single(sentence, model)
            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--execute_method", type=str, choices=["2step", "factored"], required=True)

    args = parse_args(parser)
    init_env(args, "CoVE")

    method = CoVE(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
