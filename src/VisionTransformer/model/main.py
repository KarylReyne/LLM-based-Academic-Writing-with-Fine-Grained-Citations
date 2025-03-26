import argparse
import sys
import torch as th
import numpy as np
from copy import copy
import os
import torch.distributed as dist


from utils.configuration import Configuration
from model.scripts import training
import subprocess
import fcntl
import json
import time

def update_path(cfg, scratch_path):
    source_path  = cfg['data']['hdf5_file']
    updated_path = source_path.replace('./', scratch_path)
    print(f"Updating: {source_path} -> {updated_path}")
    cfg['data']['hdf5_file'] = updated_path

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", required=True, type=str)
    parser.add_argument("-scratch", "--scratch", type=str, default="")
    parser.add_argument("-num-gpus", "--num-gpus", default=1, type=int)
    parser.add_argument("-n", "--n", default=-1, type=int)
    parser.add_argument("-load", "--load", default="", type=str)
    parser.add_argument("-port", "--port", default=29500, type=int)
    parser.add_argument("-device", "--device", default=0, type=int)
    parser.add_argument("-seed", "--seed", default=1234, type=int)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-train", "--train", action="store_true")
    mode_group.add_argument("-validate", "--validate", action="store_true")
    parser.add_argument("-float32-matmul-precision", "--float32-matmul-precision", default="highest", type=str)

    args = parser.parse_args(sys.argv[1:])

    th.set_float32_matmul_precision(args.float32_matmul_precision)

    cfg = Configuration(args.cfg)
    if args.scratch != "":
        cfg = update_path(cfg, args.scratch)
    
    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    cfg.validate = args.validate

    if args.device >= 0:
        cfg.device = args.device
        cfg.model_path = f"{cfg.model_path}.device{cfg.device}"

    if args.n >= 0:
        cfg.model_path = f"{cfg.model_path}.run{args.n}"

    num_gpus = th.cuda.device_count()
    
    if cfg.device >= num_gpus:
        cfg.device = num_gpus - 1

    if args.num_gpus > 0:
        num_gpus = args.num_gpus

    if args.train or args.validate:
        training.train(cfg, args.load if args.load != "" else None)
