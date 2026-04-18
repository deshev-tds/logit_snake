import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import TEMPLATE_ZH, TARGETS_ZH
from template_en import TEMPLATE_EN, TARGETS_EN
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result
from utils.wrapper import ModelConfiguration


class ChainOfThought(AbstractMethod):
    def __init__(self, args):
        super().__init__(args)

        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN
        self.targets = TARGETS_ZH if args.language == 'zh' else TARGETS_EN

        self.k = self.args.k

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

                # multiple chain-of-thought
                multiple_prob = []
                with ModelConfiguration(model, config_name='do_sample', value=True, enable=self.k > 1):
                    for _ in range(self.k):
                        def post_process(prob, response, output):
                            assert any(t in response for t in self.targets)
                            return prob, response, output

                        _, response, _ = model.understand(query=prompt, post_process=post_process)
                        if self.targets[1] in response:
                            prob = 1
                        elif self.targets[0] in response:
                            prob = 0
                        else:
                            prob = 0.5

                        multiple_prob.append(prob)

                prob = inverse_softmax(np.mean(multiple_prob))
            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--k", type=int, default=1)

    args = parse_args(parser)
    init_env(args, "CoT", postfix=f"{args.k}")

    method = ChainOfThought(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
