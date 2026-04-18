import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import TEMPLATE_ZH, TARGETS_ZH
from template_en import TEMPLATE_EN, TARGETS_EN
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result


class GlobalChainOfThought(AbstractMethod):
    def __init__(self, args):
        super().__init__(args)

        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN
        self.targets = TARGETS_ZH if args.language == 'zh' else TARGETS_EN

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        core_sentences = [s for s, b in zip(sentences, is_core_sentences) if b]
        core_sentences = [re.sub("^\d+\.", "", s) for s in core_sentences]
        core_sentences_str = [f"*{i + 1}* {s}" for i, s in enumerate(core_sentences)]
        core_sentences_str = "\n".join(core_sentences_str)

        placeholder = [f"*{i + 1}* [Reasoning] [Choice]" for i in range(len(core_sentences))]
        placeholder_str = "\n".join(placeholder)

        prompt = self.template
        prompt = prompt.replace('##QUERY##', core_sentences_str)
        prompt = prompt.replace('##FORMAT##', placeholder_str)

        def validate_output(prob, response, _):
            outputs = re.findall(r'\*\d+\*\s+(.*)', response)
            assert len(outputs) == len(core_sentences)
            return prob, outputs, _

        _, outputs, _ = model.understand(query=prompt, post_process=validate_output)

        pivot = 0
        for sentence, is_core_sentence in zip(sentences, is_core_sentences):
            if is_core_sentence:
                response = outputs[pivot]
                if self.targets[1] in response:
                    prob = 1
                elif self.targets[0] in response:
                    prob = 0
                else:
                    prob = 0.5

                prob = inverse_softmax(prob)
                pivot += 1
            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    args = parse_args(parser)
    init_env(args, "GlobalCoT")

    method = GlobalChainOfThought(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
