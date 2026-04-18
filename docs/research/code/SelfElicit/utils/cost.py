import numpy as np


class CostCounter:
    def __init__(self):
        self.n_generation = None
        self.gen_token_length = None

        self.reset()

    def reset(self):
        self.n_generation = 0
        self.gen_token_length = []

    def step(self, token_length):
        self.n_generation += 1
        if token_length:
            self.gen_token_length.append(token_length)

    def get_result(self):
        costs = self.n_generation, np.sum(self.gen_token_length).item()
        self.reset()
        return costs
