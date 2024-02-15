import os
import sys
import argparse
from tqdm import tqdm
import socket
import signal
import subprocess
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import _init_paths
from _init_paths import add_path, this_dir
from utils.dir_utils import export_pose_results
from recon import reconstruct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-e', required=True, type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
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
    
    add_path(os.path.join(os.path.dirname(os.path.abspath(args.cfg)), os.pardir))
    from config import cfg, update_config
    from base import Trainer, Tester
    from utils.dist_utils import reduce_tensor
    update_config(cfg, args)

    cudnn.benchmark = True
    writer_dict = {'writer': SummaryWriter(log_dir = cfg.log_dir), 'train_global_steps': 0}

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (inputs, targets, metas) in enumerate(trainer.batch_generator):
            trainer.set_lr(epoch, itr)
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # inputs
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            # targets
            for k, v in targets.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        targets[k][i] = targets[k][i].cuda(non_blocking=True)
                else:
                    targets[k] = targets[k].cuda(non_blocking=True)

            # meta infos
            metas['epoch'] = epoch
            for k, v in metas.items():
                if k != 'id' and k != 'epoch' and k != 'obj_id':
                    if isinstance(v, list):
                        for i in range(len(v)):
                            metas[k][i] = metas[k][i].cuda(non_blocking=True)
                    else:
                        metas[k] = metas[k].cuda(non_blocking=True)

            # forward
            trainer.optimizer.zero_grad()
            if cfg.task in ['hsdf_osdf_1net', 'hsdf_osdf_2net', 'hsdf_osdf_2net_pa', 'hsdf_osdf_2net_video_pa']:
                loss, sdf_results, hand_pose_results, obj_pose_results = trainer.model(inputs, targets, metas, 'train')
            elif cfg.task == 'pose_kpt':
                loss, hand_pose_results, obj_pose_results = trainer.model(inputs, targets, metas, 'train')

            # backward
            all_loss = sum(loss[k].mean() for k in loss)
            all_loss.backward()

            trainer.optimizer.step()
            torch.cuda.synchronize()

            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fs/epoch' % (trainer.tot_timer.average_time * trainer.itr_per_epoch),
                ]

            # save
            record_dict = {}
            for k, v in loss.items():
                record_dict[k] = v.detach().mean() * 1000.
            screen += ['%s: %.3f' % ('loss_' + k, v) for k, v in record_dict.items()]

            tb_writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            if itr % 10 == 0:
                trainer.logger.info(' '.join(screen))
                for k, v in record_dict.items():
                    tb_writer.add_scalar('loss_' + k, v, global_steps)
                tb_writer.add_scalar('lr', trainer.get_lr(), global_steps)
                writer_dict['train_global_steps'] = global_steps + 10

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        if (epoch % cfg.model_save_freq == 0 or epoch == cfg.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
            writer_dict['writer'].close()
        
    # test
    torch.cuda.empty_cache()
    tester = Tester(cfg.end_epoch - 1)
    tester._make_batch_generator()
    tester._make_model()

    with torch.no_grad():
        for itr, (inputs, metas) in tqdm(enumerate(tester.batch_generator)):
            # inputs
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            # meta infos
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
            export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
            export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)
            reconstruct(cfg, metas['id'], tester.model, sdf_feat, inputs, metas, hand_pose_results, obj_pose_results)
         
    
if __name__ == "__main__":
    main()
