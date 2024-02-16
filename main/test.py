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

    tester = Tester(test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    with torch.no_grad():
        for itr, (inputs, metas) in tqdm(enumerate(tester.batch_generator)):
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            for k, v in metas.items():
                if k != 'id' and k != 'obj_id':
                    if isinstance(v, list):
                        for i in range(len(v)):
                            metas[k][i] = metas[k][i].cuda(non_blocking=True)
                    else:
                        metas[k] = metas[k].cuda(non_blocking=True)

            # forward
            sdf_feat, hand_pose_results, obj_pose_results = tester.model(inputs, targets=None, metas=metas, mode='test')

            # save
            from recon import reconstruct
            export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
            export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)
            reconstruct(cfg, metas['id'], tester.model, sdf_feat, inputs, metas, hand_pose_results, obj_pose_results)


if __name__ == "__main__":
    main()
