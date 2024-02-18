import os
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from _init_paths import add_path
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
    
    add_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'models'))
    from lib.core.config import cfg, update_config
    from lib.core.base import Trainer, Tester
    update_config(args.cfg, args)

    cudnn.benchmark = True
    writer_dict = {'writer': SummaryWriter(log_dir = cfg.log_dir), 'train_global_steps': 0}

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    trainer.run(writer_dict)
        
    # test
    torch.cuda.empty_cache()
    tester = Tester(cfg.end_epoch - 1)
    tester._make_batch_generator()
    tester._make_model()
    tester.run()
         
    
if __name__ == "__main__":
    main()
