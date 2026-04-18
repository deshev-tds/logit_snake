import os
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from template_zh import TEMPLATE_ZH, TARGETS_ZH
from template_en import TEMPLATE_EN, TARGETS_EN
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from methods.common import preprocess, get_default_result
from methods.thought.judge import judge_knowledge, judge_sentence
from methods.thought.extract import extract_knowledge


class IO(AbstractMethod):
    """
    Language Models (Mostly) Know What They Know
    Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators
    """

    def __init__(self, args):
        super().__init__(args)

        self.template = TEMPLATE_ZH if args.language == 'zh' else TEMPLATE_EN
        self.targets = TARGETS_ZH if args.language == 'zh' else TARGETS_EN

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []
        history = []

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        if self.args.knowledge:
            for i in range(len(sentences)):
                if not is_core_sentences[i]:
                    sentences[i] = None

            # extract knowledge
            knowledges, entities = extract_knowledge(sentences,
                                                     method=self.args.knowledge,
                                                     model=model,
                                                     sample_id=kwargs['sample_id'],
                                                     knowledge_cache=os.path.join(self.args.save_folder, 'knowledge'),
                                                     language=self.args.language)

        prev_sent = []
        for sid, (sentence, is_core_sentence) in enumerate(zip(sentences, is_core_sentences)):
            if is_core_sentence:
                if self.args.knowledge:
                    prob, response = judge_knowledge(knowledges[sid], model, self.args.language)
                else:
                    prompt = self.template
                    if prev_sent and not args.use_history:
                        if self.args.language == 'zh':
                            prompt = "上下文：" + "。".join(prev_sent) + "\n" + prompt
                        elif self.args.use_context:
                            prompt = "Context: " + ". ".join(prev_sent) + "\n" + prompt
                    prompt = prompt.replace('##TARGETS##', '\n'.join(self.targets))
                    prompt = prompt.replace('##QUERY##', sentence)

                    prob, response, _ = model.understand(query=prompt,
                                                         history=history if args.use_history else [],
                                                         targets=self.targets)

                    response = self.targets[prob.argmax(axis=0).item()]
                    prob = prob.float().cpu().numpy()
                    history.append((prompt, response))
                    prev_sent.append(sentence)

            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--knowledge", type=str)
    parser.add_argument("--use_context", action="store_true")

    args = parse_args(parser)
    if args.use_context:
        postfix = "context"
    elif args.knowledge:
        postfix = str(args.knowledge)
    else:
        postfix = None
    init_env(args, "IO", postfix=postfix)

    method = IO(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
