import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils import io_tools
from utils.answering import DEFAULT_MODEL_CONFIGS, BaseAnsweringModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mllm_name", type=str, required=True, choices=set(DEFAULT_MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", type=str, required=True, choices={'gpt4', 'mc', 'greedy', 'prefix'})
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)

    load_path = f'{ROOT}/Results/{args.mllm_name}/{args.mllm_name}_{args.mode}_score.json'

    score = io_tools.load_json(load_path)
    BaseAnsweringModel.print_score(score)
