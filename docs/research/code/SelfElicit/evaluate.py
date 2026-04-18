import os
import pickle
from utils.metrics import get_metrics
from data.dataloader import load_dataset
from utils.llm import load_model
from methods import AbstractMethod


class Evaluator:
    def __init__(self, args):
        datasets = load_dataset(args.data_path, args.verbose)
        self.dataset = datasets['test']
        self.model = load_model(args.__dict__)

        self.args = args

    def evaluate(self, method: AbstractMethod):
        probs, responses, costs = [], [], [0, 0]
        for sid, sample in enumerate(self.dataset):
            print("*" * 30 + f"[{sid}/{len(self.dataset)}]" + "*" * 30)

            # read from cache
            if self.args.use_cache and os.path.exists(os.path.join(self.args.cache_folder, f"{sid}.pkl")):
                with open(os.path.join(self.args.cache_folder, f"{sid}.pkl"), "rb") as f:
                    data = pickle.load(f)
                    prob, response = data['prob'], data['response']
                    cost = data['cost']

            else:
                if self.args.use_claim:
                    sample['sentences'] = sample['claims']
                    sample['references'] = sample['claim_retrievals'] if 'claim_retrievals' in sample else []
                else:
                    sample['sentences'] = sample['sentences']
                    sample['references'] = sample['sentence_retrievals'] if 'sentence_retrievals' in sample else []

                self.model.counter.reset()
                prob, response = method(sample, self.model, sample_id=sid)
                cost = self.model.counter.get_result()

                if self.args.use_cache:
                    with open(os.path.join(self.args.cache_folder, f"{sid}.pkl"), "wb") as fp:
                        pickle.dump({'prob': prob, 'response': response, 'cost': cost}, fp)

            probs.append(prob)
            responses.append(response)
            costs[0] += cost[0]
            costs[1] += cost[1]

        # metrics
        print(f"Total costs: {costs[0]} generations, {costs[1]} tokens")
        labels = [sample['claim_labels'] if self.args.use_claim else sample['sentence_labels'] for sample in self.dataset]
        sample_labels = [sample['label'] for sample in self.dataset]
        with open(os.path.join(self.args.save_folder, f"output.pkl"), "wb") as fp:
            pickle.dump({'probs': probs, 'responses': responses, 'costs': costs}, fp)

        metric = get_metrics(probs=probs,
                             labels=labels,
                             sample_labels=sample_labels,
                             strategies=self.args.strategies,
                             aggregate=self.args.aggregate,
                             penalty=self.args.penalty)
        return metric
