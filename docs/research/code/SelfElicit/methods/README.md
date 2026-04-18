Here is a template of baseline methods

```python
from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics

TEMPLATE = """\
Judge whether the following description is correct.
#Description: ##QUERY##
#Is the description: ##TARGETS##
#The description is:
"""
TARGETS = ['A.Correct', 'B.Wrong', 'C.None of above']


class YourMethod(AbstractMethod):
    def __call__(self, sample, model, **kwargs):
        """
        @inputs
            sample: Dict with `sentences`(N sentences to be judged) and `references`(N reference information blocks with respect to sentence. Each block has several texts).
            model: An instance of class LLM
        @returns
            probs: The un-normalized scores for positive/negative/neutral. np.ndarray with shape (N, 3).
            responses: N responses
        @raise
            Any exception. Will retry if any exception has been raised.
        """
        probs, responses = [], []
        for sentence in sample['sentences']:
            query = TEMPLATE.replace('##QUERY##', sentence)
            query = query.replace('##TARGETS##', " ".join(TARGETS))

            prob, response = model.understand(query=query, history=[], targets=TARGETS)
            probs.append(prob.float().cpu().numpy())
            responses.append(response)

            if response[0] == 'C':
                raise ValueError(f"You can raise any exception")

        return probs, responses


if __name__ == '__main__':
    parser.add_argument("--argument", help="Your extra arguments")

    args = parse_args(parser)
    init_env(args, "YourMethod", postfix="YourPostFix")

    method = YourMethod(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
```