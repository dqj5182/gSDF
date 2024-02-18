import os
import sys
import argparse
import yaml
from tqdm import tqdm
import torch
from loguru import logger
import _init_paths
from _init_paths import add_path, this_dir
from lib.utils.dir_utils import export_pose_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='experiment configure file name')
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
    parser.add_argument('--checkpoint', type=str, default='', help='model path for evaluation')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    task = args.cfg.split('/')[-3]
    add_path(os.path.join('../lib', 'models'))
    from lib.core.config import cfg, update_config
    update_config(args.cfg, args, mode='test')
    from lib.core.base import Tester

    if cfg.MODEL.weight_path != '':
        test_epoch = int(cfg.MODEL.weight_path.split('/')[-1].split('snapshot_')[-1].split('.pth.tar')[0])
    else:
        test_epoch = cfg.TRAIN.end_epoch - 1

    tester = Tester(args, test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    tester.run()


if __name__ == "__main__":
    main()
