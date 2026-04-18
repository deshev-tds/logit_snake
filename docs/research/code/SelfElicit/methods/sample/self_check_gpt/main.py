import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import TEMPLATE_ZH, TARGETS_ZH
from template_en import TEMPLATE_EN, TARGETS_EN
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from utils.wrapper import ModelConfiguration
from methods.common import preprocess, get_default_result


class SelfCheckGPT(AbstractMethod):
    """
    SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models
    """

    def __init__(self, args):
        super().__init__(args)

        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN
        self.targets = TARGETS_ZH if args.language == 'zh' else TARGETS_EN

        self.n_stochastic_responses = args.n_stochastic_responses
        self.method = args.method

    def get_stochastic_responses(self, question, model):
        responses = []
        for _ in range(self.n_stochastic_responses):
            with ModelConfiguration(model, config_name='do_sample', value=True, enable=True):
                _, response, _ = model.understand(question)
            responses.append(response.replace("\n", ""))
        return responses

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        question = sample['question']
        stochastic_responses = self.get_stochastic_responses(question, model)

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                        cls_cache=self.args.cls_cache,
                                        context_cache=self.args.context_cache,
                                        language=self.args.language)

        for sentence, is_core_sentence in zip(sentences, is_core_sentences):
            if is_core_sentences:
                scores, resp = [], []
                for ref in stochastic_responses:
                    query = self.template
                    query = query.replace("##CONTEXT##", ref)
                    query = query.replace("##SENTENCE##", sentence)
                    _, response, _ = model.understand(query=query)

                    response = response.lower()

                    if self.targets[0] in response and f"not {self.targets[0]}" not in response:
                        scores.append(1)
                    elif self.targets[1] in response and f"not {self.targets[0]}" not in response:
                        scores.append(0)
                    else:
                        scores.append(0.5)
                    resp.append(response)

                scores = np.mean(scores)
                prob = inverse_softmax(scores)
                response = "\t".join(resp)

            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--n_stochastic_responses", default=20, type=int)
    parser.add_argument("--method", default='prompt', choices=['prompt'])
    args = parse_args(parser)

    init_env(args, "SelfCheckGPT", postfix=args.method)

    method = SelfCheckGPT(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
