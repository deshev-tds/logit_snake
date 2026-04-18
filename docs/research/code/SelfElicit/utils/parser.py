import argparse

parser = argparse.ArgumentParser()

# model config
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--server", type=str, help="API server")

# generation config
parser.add_argument("--do_sample", action='store_true')
parser.add_argument("--use_history", action='store_true')
parser.add_argument("--max_new_tokens", type=int, default=8192)
parser.add_argument("--top_k", type=float)
parser.add_argument("--top_p", type=float)
parser.add_argument("--repetition_penalty", type=float)
parser.add_argument("--max_attempts", type=int, default=5, help="Max retry attempts when LLMs output undesired outputs.")

# data
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--info", type=str, choices=['claim', 'sentence'], required=True, help="Detect hallucination on each sentence or claim.")
parser.add_argument("--language", type=str, choices=['zh', 'en'], default='zh')

# metrics
parser.add_argument("--strategies", type=str, nargs='+', default=['ignore'], choices=['ignore', 'neg', 'pos'], help="How to treat `Not sure` result.")
parser.add_argument("--aggregate", type=str, default='max', choices=['max', 'prod', 'mean'], help="Aggregate sentence/claim-wise result into query-wise result.")
parser.add_argument("--penalty", type=int, help="Penalize if excess claims/sentences")

# save
parser.add_argument("--save_path", type=str, default="./saves")
parser.add_argument("--use_cache", action='store_true', help="Save result for each sample in order to restart from checkpoints after interruptions.")

# other
parser.add_argument("--seed", type=int, default=3317)
parser.add_argument("--verbose", action='store_true')

# preprocess
parser.add_argument("--cls_cache", type=str, help="Set the cache path to enable classification")
parser.add_argument("--context_cache", type=str, help="Set the cache path to enable decontextualisation")


def parse_args(parser):
    args = parser.parse_args()

    if args.info == 'claim':
        args.use_claim = True
    else:
        args.use_claim = False

    args.model_path = args.model_path.rstrip('/')
    args.model_name = args.model_path.split('/')[-1].lower().replace('-', '_')
    args.data_name = args.data_path.rstrip('/').split('/')[-1]

    return args
