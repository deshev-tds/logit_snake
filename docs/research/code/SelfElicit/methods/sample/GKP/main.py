import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import *
from template_en import *
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics
from methods.common import preprocess, get_default_result


class GKP(AbstractMethod):
    """
    Generated Knowledge Prompting for Commonsense Reasoning
    """

    def __init__(self, args):
        super().__init__(args)

        self.knowledge_template = KNOWLEDGE_PROMPTING_ZH if args.language == 'zh' else KNOWLEDGE_PROMPTING_EN
        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN
        self.targets = TARGETS_ZH if args.language == 'zh' else TARGETS_EN

    def __call__(self, sample, model, **kwargs):
        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        probs, responses = [], []
        for sid, (sentence, is_core_sentence) in enumerate(zip(sentences, is_core_sentences)):
            if is_core_sentence:
                prompt = self.knowledge_template
                prompt = prompt.replace('##QUERY##', sentence)
                _, response, _ = model.understand(query=prompt)
                history = (prompt, response)

                prompt = self.template
                prompt = prompt.replace('##TARGETS##', '\n'.join(self.targets))
                prompt = prompt.replace('##QUERY##', sentence)

                prob, response, _ = model.understand(query=prompt,
                                                     history=[history],
                                                     targets=self.targets)
                prob = prob.float().cpu().numpy()

            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    args = parse_args(parser)
    init_env(args, "GKP")

    method = GKP(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
