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
    for epoch in range(trainer.start_epoch, cfg.TRAIN.end_epoch):
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
            sdf_results, hand_pose_results, obj_pose_results, inter_results, processed_gt = trainer.model(inputs, targets, metas, 'train')

            # loss
            loss_l1 = torch.nn.L1Loss(reduction='sum')
            loss_l2 = torch.nn.MSELoss()
            loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

            loss = {}
            loss['hand_sdf'] = cfg.TRAIN.hand_sdf_weight * loss_l1(sdf_results['hand'] * inter_results['mask_hand'], processed_gt['sdf_gt_hand'] * inter_results['mask_hand']) / inter_results['mask_hand'].sum()
            loss['obj_sdf'] = cfg.TRAIN.obj_sdf_weight * loss_l1(sdf_results['obj'] * inter_results['mask_obj'], processed_gt['sdf_gt_obj'] * inter_results['mask_obj']) / inter_results['mask_obj'].sum()
            
            loss['volume_joint'] = cfg.TRAIN.volume_weight * loss_l2(obj_pose_results['center'], targets['obj_center_3d'].unsqueeze(1))

            if cfg.MODEL.obj_rot and cfg.TRAIN.corner_weight > 0:
                loss['obj_corners'] = cfg.TRAIN.corner_weight * loss_l2(obj_pose_results['corners'], targets['obj_corners_3d'])

            if sdf_results['cls_hand'] is not None:
                if metas['epoch'] >= cfg.TRAIN.sdf_add_epoch:
                    loss['hand_cls'] = cfg.TRAIN.hand_cls_weight * loss_ce(sdf_results['cls_hand'], processed_gt['cls_data'])
                else:
                    loss['hand_cls'] = 0. * loss_ce(sdf_results['cls_hand'], processed_gt['cls_data'])

            # backward
            all_loss = sum(loss[k].mean() for k in loss)
            all_loss.backward()

            trainer.optimizer.step()
            torch.cuda.synchronize()

            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.TRAIN.end_epoch, itr, trainer.itr_per_epoch),
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
        
        if (epoch % cfg.model_save_freq == 0 or epoch == cfg.TRAIN.end_epoch - 1):
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
            from recon import reconstruct
            export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
            export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)
            reconstruct(cfg, metas['id'], tester.model, sdf_feat, inputs, metas, hand_pose_results, obj_pose_results)
         
    
if __name__ == "__main__":
    main()
