import os
from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics
from utils.llm import NLIModel
from methods.common import preprocess
from methods.elicit.extract import extract_knowledge
from methods.elicit.judge import *
from methods.elicit.self_kg import judge_with_self_kg


class Thought(AbstractMethod):
    def __init__(self, args):
        super().__init__(args)

        self.nli_model = NLIModel(args.nli_model) if args.nli_model else None

    def judge_loop(self, candidates, method, model, language, **kwargs):
        context = []
        probs = []
        responses = []

        for i, candidate in enumerate(candidates):
            if candidate:
                if self.args.judge_with_context:
                    kwargs['context'] = context

                else:
                    kwargs['context'] = None

                prob, response = method(candidate, model, language, **kwargs)
                if isinstance(candidate, list):
                    context.extend(candidate)
                else:
                    context.append(candidate)

            else:
                prob = inverse_softmax(0)  # correct
                response = "Regard as correct."

            probs.append(prob)
            responses.append(response)

        return probs, responses

    def merge_responses(self, orig, mergee, stage, sep='\t'):
        assert len(orig) == len(mergee)
        for i in range(len(orig)):
            orig[i] += f"{sep}[{stage}]{mergee[i]}"
        return orig

    def __call__(self, sample, model, **kwargs):
        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        responses = [[]] * len(sentences)
        for i in range(len(sentences)):
            if not is_core_sentences[i]:
                sentences[i] = None

        # extract knowledge
        knowledge_cache = self.args.knowledge_cache or os.path.join(self.args.save_folder, 'knowledge')
        knowledges, entities = extract_knowledge(sentences,
                                                 method=self.args.knowledge,
                                                 model=model,
                                                 sample_id=kwargs['sample_id'],
                                                 knowledge_cache=knowledge_cache,
                                                 language=self.args.language)

        # judge
        if self.args.judge == 'sentence':
            probs, resp = self.judge_loop(sentences, judge_sentence, model, language=self.args.language)
        elif self.args.judge == 'knowledge':
            probs, resp = self.judge_loop(knowledges, judge_knowledge, model, entities=entities, language=self.args.language)
        elif self.args.judge == 'knowledge_question':
            probs, resp = self.judge_loop(knowledges, judge_knowledge_question, model, language=self.args.language)
        elif self.args.judge == 'self_kg':
            probs, resp = judge_with_self_kg(knowledges, entities, model, self.nli_model, language=self.args.language)
        else:
            raise ValueError()

        responses = self.merge_responses(responses, resp, stage='judge')

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--postfix", type=str)

    parser.add_argument("--knowledge", type=str)
    parser.add_argument("--knowledge_cache", type=str)

    parser.add_argument("--judge", type=str)
    parser.add_argument("--judge_with_context", action='store_true')

    parser.add_argument("--nli_model", type=str)

    args = parse_args(parser)
    init_env(args, "elicit", postfix=args.postfix)

    method = Thought(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
