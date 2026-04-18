import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import TEMPLATE_ZH
from template_en import TEMPLATE_EN
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result


class ConfidenceScore(AbstractMethod):
    """
    Factual Confidence of LLMs: on Reliability and Robustness of  Current Estimators
    """

    def __init__(self, args):
        super().__init__(args)

        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        for sentence, is_core_sentence in zip(sentences, is_core_sentences):
            if is_core_sentence:
                prompt = self.template
                prompt = prompt.replace('##QUERY##', sentence)

                _, response, _ = model.understand(query=prompt)

                pattern = r'[0-9]+\.[0-9]+'
                match = re.findall(pattern, response)
                if match:
                    prob = 1 - float(match[0]) / 10.0
                    print(prob)
                else:
                    prob = 0

                prob = inverse_softmax(prob)
            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    args = parse_args(parser)
    init_env(args, "ConfidenceScore")

    method = ConfidenceScore(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
