from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from template import *
import numpy as np
from methods.retrieval.FActScore.npm import NPM  # use absolute path here


class FActScore(AbstractMethod):
    """
    FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
    """

    def __init__(self, args):
        super().__init__(args)

        self.npm_model = NPM(args.npm_model_path, args.device_map)

    def get_score(self, sentence, references, model):
        references = references[:self.args.n_retrieval]

        prompt = JUDGE_TEMPLATE_RETRIEVAL
        context = []
        for rid, ref in enumerate(references):
            ref = ref.replace('\n', '')
            context.append(f"参考材料{rid + 1}：{ref}")
        prompt = prompt.replace("##CONTEXT##", "\n".join(context))

        prompt = prompt.replace("##SENTENCE##", sentence.replace("\n", ""))
        logits, response, _ = model.understand(query=prompt,
                                               history=[],
                                               targets=JUDGE_TARGETS)

        if logits[0] <= logits[1]:
            is_supported = False
        else:  # if is_supported=True
            np_prob = self.npm_model.get_probability(sentence, references)
            is_supported = np_prob > self.args.npm_thres

        return is_supported, response

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []
        for sentence, references in zip(sample['sentences'], sample['references']):
            is_supported, response = self.get_score(sentence, references, model)

            probs.append(inverse_softmax(1 - is_supported))
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--n_retrieval", type=int, default=5)
    parser.add_argument("--npm_model_path", type=str)
    parser.add_argument("--npm_thres", type=float, default=0.005)

    args = parse_args(parser)
    init_env(args, "FActScore")

    method = FActScore(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
