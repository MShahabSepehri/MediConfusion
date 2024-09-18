import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import shutil
import argparse
import subprocess
from tqdm import tqdm
from utils import io_tools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='data/images')
    parser.add_argument("--image_dict_path", type=str, default='data/image_dict.json')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    image_dict = io_tools.load_json(args.image_dict_path)

    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    for key in tqdm(image_dict.keys()):
        sample = image_dict.get(key)
        local = sample.get('local')
        roco_id = key.split('/')[-1].replace('.jpg', '')
        link = sample.get('dlink')
        file_name = sample.get('file_name')
        pmc_name = link.split('/')[-1].replace('.tar.gz', '')
        subprocess.call(['wget', '-q', link, '-P', f'{args.save_path}/'])
        subprocess.call(['tar', '-xzf', f'{args.save_path}/{pmc_name}.tar.gz', '-C', args.save_path])
        shutil.copy(f'{args.save_path}/{pmc_name}/{file_name}', f'{args.save_path}/{local}.jpg')
        shutil.rmtree(f'{args.save_path}/{pmc_name}')
        os.remove(f'{args.save_path}/{pmc_name}.tar.gz')