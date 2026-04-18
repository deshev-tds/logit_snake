import numpy as np
from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from template_zh import *
from template_en import *
from methods.common import preprocess, get_default_result


class SelfCheck(AbstractMethod):
    """
    SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning
    """
    def __init__(self, args):
        super().__init__(args)

        self.step = '步骤' if args.language == 'zh' else 'Step '
    
        self.target_extraction_template = TARGET_EXTRACTION_TEMPLATE_ZH if args.language == 'zh' else TARGET_EXTRACTION_TEMPLATE_EN

        self.information_collection_first_template = INFORMATION_COLLECTION_FIRST_TEMPLATE_ZH if args.language == 'zh' else INFORMATION_COLLECTION_FIRST_TEMPLATE_EN
        self.information_collection_template = INFORMATION_COLLECTION_TEMPLATE_ZH if args.language == 'zh' else INFORMATION_COLLECTION_TEMPLATE_EN

        self.rerunning_first_template = RERUNNING_FIRST_TEMPLATE_ZH if args.language == 'zh' else RERUNNING_FIRST_TEMPLATE_EN
        self.rerunning_template = RERUNNING_TEMPLATE_ZH if args.language == 'zh' else RERUNNING_TEMPLATE_EN

        self.compare_template = COMPARE_TEMPLATE_ZH if args.language == 'zh' else COMPARE_TEMPLATE_EN
        self.compare_targets = COMPARE_TARGETS_ZH if args.language == 'zh' else COMPARE_TARGETS_EN

    def __call__(self, sample, model, **kwargs):
        probs, responses = [], []

        all_steps = [f"[{self.step}{1 + i}] " + s.replace('\n', '') for i, s in enumerate(sample['sentences'])]

        sentences, is_core_sentences = preprocess(sample["sentences"], model, kwargs["sample_id"],
                                                  cls_cache=self.args.cls_cache,
                                                  context_cache=self.args.context_cache,
                                                  language=self.args.language)

        for sid, (sentence, is_core_sentence) in enumerate(zip(sentences, is_core_sentences)):
            if is_core_sentence:
                # target extraction
                prompt = self.target_extraction_template
                prompt = prompt.replace("##QUESTION##", sample['question'])
                prompt = prompt.replace("##STEPS##", '\n'.join(all_steps))
                prompt = prompt.replace("##STEP##", f"[{self.step}{sid + 1}] {sentence}")
                _, target, _ = model.understand(query=prompt)

                # information collection
                prompt = self.information_collection_first_template if sid == 0 else self.information_collection_template
                prompt = prompt.replace("##QUESTION##", sample['question'])
                prompt = prompt.replace("##STEPS##", '\n'.join(all_steps[:sid]))
                prompt = prompt.replace("##STEP##", f"“{all_steps[sid]}”")
                _, information, _ = model.understand(query=prompt)

                # rerun
                prompt = self.rerunning_first_template if sid == 0 else self.rerunning_template
                prompt = prompt.replace("##STEPS##", '\n'.join(all_steps[:sid]))
                prompt = prompt.replace("##INFORMATION##", information)
                prompt = prompt.replace("##TARGET##", target)
                prompt = prompt.replace("##STEPID##", str(sid + 1))
                _, rerun, _ = model.understand(query=prompt)
                rerun = rerun.replace("\n", "")

                # compare
                prompt = self.compare_template
                prompt = prompt.replace("##ANSWER_ORIG##", sentence)
                prompt = prompt.replace("##ANSWER_RERUN##", rerun)
                _, compare, _ = model.understand(query=prompt)

                # summarize
                if self.compare_targets[0] in compare:
                    prob = inverse_softmax(0)
                    response = self.compare_targets[0]
                elif self.compare_targets[1] in compare:
                    prob = inverse_softmax(1)
                    response = self.compare_targets[1]
                else:
                    prob = inverse_softmax(0)
                    response = self.compare_targets[2]
                response = f"Rerun step {rerun}, {response} from original step."

            else:
                prob, response = get_default_result()

            probs.append(prob)
            responses.append(response)

        return np.stack(probs), responses


if __name__ == '__main__':
    args = parse_args(parser)
    init_env(args, "SelfCheck")

    method = SelfCheck(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
