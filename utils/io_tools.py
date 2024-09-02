import os
import json
import yaml
import torch
import pickle
import pathlib
import importlib


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_config_from_yaml(path):
    config_file = pathlib.Path(path)
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
        return d
    else:
        raise ValueError('Config file does not exist.')


def str2int(s):
    return int.from_bytes(s.encode(), 'little') % (2 ** 32 - 1)


def save_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        tmp = pickle.load(file)
    return tmp


def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_root(file, num_returns=1):
    tmp = pathlib.Path(file)
    for _ in range(num_returns):
        tmp = tmp.parent.resolve()
    return tmp


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def modify_json(data, path):
    old_data = load_json(path)
    new_keys = 0
    for key in data.keys():
        if key in old_data:
            continue
        old_data[key] = data.get(key)
        new_keys += 1
    save_json(old_data, path)
    return new_keys

def load_resume_dict(path):
    if path is None:
        return {}
    return load_json(path)