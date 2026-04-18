import re
import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from template_zh import *
from template_en import *
from utils.wrapper import ModelConfiguration
from methods.common import preprocess, get_default_result


class ChatProtect(AbstractMethod):
    """
    Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation
    """

    def __init__(self, args):
        super().__init__(args)

        if self.args.ie_agg == 'max':
            self.agg_fn = np.max
        elif self.args.ie_agg == 'mean':
            self.agg_fn = np.mean
        else:
            raise NotImplementedError

        self.information_extraction_fewshots = INFORMATION_EXTRACTION_FEW_SHOTS_ZH if args.language == 'zh' else INFORMATION_EXTRACTION_FEW_SHOTS_EN
        self.information_extraction_fewshots = self.information_extraction_fewshots[:self.args.n_information_extraction_few_shots]
        self.information_extraction_template = INFORMATION_EXTRACTION_TEMPLATE_ZH if args.language == 'zh' else INFORMATION_EXTRACTION_TEMPLATE_EN

        self.statement_fewshots = STATEMENT_FEW_SHOTS_ZH if args.language == 'zh' else STATEMENT_FEW_SHOTS_EN
        self.statement_first_template = STATEMENT_FIRST_TEMPLATE_ZH if args.language == 'zh' else STATEMENT_FIRST_TEMPLATE_EN
        self.statement_template = STATEMENT_TEMPLATE_ZH if args.language == 'zh' else STATEMENT_TEMPLATE_EN

        self.explain_first_template = EXPLAIN_FIRST_TEMPLATE_ZH if args.language == 'zh' else EXPLAIN_FIRST_TEMPLATE_EN
        self.explain_template = EXPLAIN_TEMPLATE_ZH if args.language == 'zh' else EXPLAIN_TEMPLATE_EN

        self.consistent_template = CONSISTENT_TEMPLATE_ZH if args.language == 'zh' else CONSISTENT_TEMPLATE_EN
        self.consistent_targets = CONSISTENT_TARGET_ZH if args.language == 'zh' else CONSISTENT_TARGET_EN

    def information_extracion(self, sentence, model):
        few_shot = []
        for cid, case in enumerate(self.information_extraction_fewshots):
            if args.language == 'zh':
                shot = f"例子{cid + 1}：\n"
                shot += f"句子：{case['sentence']}\n"
                shot += f"三元组：\n"
            else:
                shot = f"Example {cid + 1}：\n"
                shot += f"Sentence: {case['sentence']}\n"
                shot += f"Triples: \n"
            shot += "\n".join(case['tuples']) + "\n"
            few_shot.append(shot)

        prompt = self.information_extraction_template
        prompt = prompt.replace("##FEW_SHOTS##", "\n".join(few_shot))
        prompt = prompt.replace("##SENTENCE##", sentence)

        def post_process(prob, response, output):
            if self.args.language == 'zh':
                response = list(filter(lambda line: len(line.split('，')) == 3, response.split("\n")))
            else:
                response = list(filter(lambda line: len(line.split(',')) == 3, response.split("\n")))

            triples = list(set(response))
            return prob, triples, output

        _, triples, _ = model.understand(query=prompt, post_process=post_process)
        return triples

    def statement_completion(self, question, sentences, sid, subject, predicate, model):
        few_shot = []
        for case in self.statement_fewshots:
            shot = self.statement_template
            shot = shot.replace("##QUESTION##", case['question'])
            shot = shot.replace("##STEPS##", case['steps'])
            shot = shot.replace("##SUBJECT##", case['subject'])
            shot = shot.replace("##PREDICATE##", case['predicate'])
            few_shot.append((shot, case['statement']))

        prompt = self.statement_first_template if sid == 0 else self.statement_template
        prompt = prompt.replace("##QUESTION##", question)
        prompt = prompt.replace("##STEPS##", "\n".join(sentences[:sid]))
        prompt = prompt.replace("##SUBJECT##", subject)
        prompt = prompt.replace("##PREDICATE##", predicate)
        _, new_sentence, _ = model.understand(query=prompt, history=few_shot)
        return new_sentence.replace("\n", "")

    def consistent_score(self, explanation, model):
        with ModelConfiguration(model, config_name='do_sample', value=True):
            tuple_score = []
            for cid in range(self.args.n_consistent_checks):
                prompt = self.consistent_template
                prompt = prompt.replace("##TARGETS##", "、".join([f"“{t}”" for t in self.consistent_targets]))
                _, consistent, _ = model.understand(query=prompt,
                                                    history=[explanation])
                if self.consistent_targets[0] in consistent and self.consistent_targets[1] not in consistent:
                    tuple_score.append(False)
                else:
                    tuple_score.append(True)

        tuple_score = np.mean(tuple_score).item()
        return tuple_score

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        for sid, (sentence, is_core_sentence) in enumerate(zip(sentences, is_core_sentences)):
            if is_core_sentence:
                # information extraction
                triples = self.information_extracion(sentence, model)

                prob = []
                if not triples:
                    prob_agg = 0.
                else:
                    for triple in triples:
                        triple = re.sub(r'[（）()]', '', triple.strip())
                        subject, predicate, object = triple.split('，') if self.args.language == 'zh' else triple.split(',')

                        # statement completion
                        new_sentence = self.statement_completion(sample["question"], sample["sentences"], sid,
                                                                 subject, predicate, model)

                        # explain
                        if sentence == new_sentence:
                            tuple_score = True
                        else:
                            prompt = self.explain_first_template if sid == 0 else self.explain_template
                            prompt = prompt.replace("##QUESTION##", sample['question'])
                            prompt = prompt.replace("##STEPS##", "\n".join(sample['sentences'][:sid]))
                            prompt = prompt.replace("##SENTENCE_A##", sentence)
                            prompt = prompt.replace("##SENTENCE_B##", new_sentence)
                            _, explanation, _ = model.understand(query=prompt)

                            # consistent
                            tuple_score = self.consistent_score((prompt, explanation), model)

                        prob.append(1 - tuple_score)

                    prob_agg = self.agg_fn(prob).item()

                prob = inverse_softmax(prob_agg)
                response = f"Extract {len(triples)} triples with scores: {prob}. Aggregated score: {prob_agg}"

            else:
                prob, response = get_default_result()
            
            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    parser.add_argument("--n_information_extraction_few_shots", default=3)
    parser.add_argument("--n_consistent_checks", default=10)
    parser.add_argument("--ie_agg", default='max', choices=['max', 'mean'])

    args = parse_args(parser)
    init_env(args, "ChatProtect", postfix=args.ie_agg)

    method = ChatProtect(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
