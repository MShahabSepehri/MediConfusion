import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils import io_tools
from utils.answering import ANSWERING_CLASS_DICT, DEFAULT_MODEL_CONFIGS 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr", type=int, default=3)
    parser.add_argument("--mllm_name", type=str, required=True, choices=set(DEFAULT_MODEL_CONFIGS.keys()))
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--model_args_path", type=str, default=None)
    parser.add_argument("--local_image_address", type=bool, default=True)
    parser.add_argument("--data_path", type=str, default='./data/images')
    parser.add_argument("--mode", type=str, required=True, choices={'gpt4', 'mc', 'greedy', 'prefix'})
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    if args.model_args_path is None:
        args.model_args_path = DEFAULT_MODEL_CONFIGS.get(args.mllm_name)

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)

    save_path = f'{ROOT}/Results/'

    answering_class = ANSWERING_CLASS_DICT.get(args.mllm_name)

    ans_obj = answering_class(args.model_args_path, args.mode, args.data_path, args.local_image_address, args.tr, args.device)
    ans_obj.evaluate(args.resume_path, save_path)
