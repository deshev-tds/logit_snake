import os
import re
import sys
import json
import torch
import random
import numpy as np
from typing import Dict


def init_env(args, method_name, postfix=None):
    """
    :param args: Arguments from command line
    :param method_name: Name of the method
    :param postfix: postfix to append at the end of the save folder
    """

    class Logger:
        def __init__(self, file_name="log.txt"):
            self.terminal = sys.stdout
            self.log = open(file_name, "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # run name
    arg_str = [args.model_name, args.info]
    if args.use_history:
        arg_str.append('history')
    if args.do_sample:
        arg_str.append('sample')
    if isinstance(postfix, list):
        arg_str.extend(postfix)
    elif isinstance(postfix, str):
        arg_str.append(postfix)

    # save folder
    args.save_folder = os.path.join("./saves", method_name, args.data_name, '-'.join(arg_str))
    args.cache_folder = os.path.join(args.save_folder, 'cache')
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.cache_folder, exist_ok=True)

    # set logger
    sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(args)


def save_metrics(path: str, metrics: Dict):
    path = os.path.join(path, "metrics.json")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=4)


def inverse_softmax(p, eps=1e-8):
    # a trick. `softmax(inverse_softmax(x)) == [1 - x, x, 0]`
    return np.array([0, np.log(p / (1 - p + eps)), -np.inf])


def match(word, text, language):
    assert language in ['zh', 'en']
    if language == 'zh':
        return word in text
    else:
        return re.findall(fr"\b{word}\b", text, re.IGNORECASE)
