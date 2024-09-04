import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils import io_tools
# from BBP.utils.vlms import llava
from utils.answering import ANSWERING_CLASS_DICT, DEFAULT_MODEL_CONFIGS 

ROOT = io_tools.get_root(__file__, 2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr", type=int, default=3)
    parser.add_argument("--vlm_name", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--model_args_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='/data/datasets/roco-dataset/data')
    parser.add_argument("--mode", type=str, required=True, choices={'gpt4', 'mc', 'greedy', 'prefix'})
    args = parser.parse_args()

    if args.model_args_path is None:
        args.model_args_path = DEFAULT_MODEL_CONFIGS.get(args.vlm_name)

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)

    save_path = f'{ROOT}/Results/{args.vlm_name}_{args.mode}.json'

    answering_class = ANSWERING_CLASS_DICT.get(args.vlm_name)

    ans_obj = answering_class(args.model_args_path, args.mode, args.data_path, args.tr)
    ans_obj.evaluate(args.resume_path, save_path)
