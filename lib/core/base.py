import os
import os.path as osp
import math
import glob
import tqdm
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from lib.utils.timer import Timer
from lib.core.config import cfg
from model import get_model
from lib.utils.train_utils import get_dataloader, get_optimizer, train_setup


class BaseTrainer:
    def __init__(self, args, load_dir, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))
        # loss
        self.loss = prepare_criterion()
        # model
        self.logger.info("Creating graph and optimizer...")
        self.model, checkpoint = prepare_network(args, load_dir, is_train=True)
        self.model, self.optimizer, self.start_epoch = train_setup(self.model, checkpoint, self.logger)

class BaseTester:
    def __init__(self, args, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))
        # model
        self.model, _ = prepare_network(args, load_dir, is_train=False)


class Trainer(BaseTrainer):
    def __init__(self, args, load_dir):
        super(Trainer, self).__init__(args=args, load_dir=load_dir, log_name='train_logs.txt')

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def set_lr(self, epoch, iter_num):
        if epoch < cfg.TRAIN.warm_up_epoch:
            cur_lr = cfg.TRAIN.lr / cfg.TRAIN.warm_up_epoch * epoch
        else:
            cur_lr = cfg.TRAIN.lr
            if cfg.TRAIN.lr_dec_style == 'step':
                for i in range(len(cfg.TRAIN.lr_dec_epoch)):
                    if epoch >= cfg.TRAIN.lr_dec_epoch[i]:
                        cur_lr = cur_lr * cfg.TRAIN.lr_dec_factor

            elif cfg.TRAIN.lr_dec_style == 'cosine':
                total_iters = cfg.TRAIN.end_epoch * len(self.batch_generator)
                warmup_iters = cfg.TRAIN.warm_up_epoch * len(self.batch_generator)
                cur_iter = epoch * len(self.batch_generator) + iter_num + 1
                cur_lr = 0.5 * (1 + np.cos(((cur_iter - warmup_iters) * np.pi) / (total_iters - warmup_iters))) * cfg.TRAIN.lr

            self.optimizer.param_groups[0]['lr'] = cur_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def run(self, args, writer_dict):
        self.batch_generator, self.itr_per_epoch = get_dataloader(cfg.DATASET.trainset_3d, cfg.DATASET.trainset_3d_split, True, logger=self.logger)
        # self._make_model(args)

        # train
        for epoch in range(self.start_epoch, cfg.TRAIN.end_epoch):
            self.tot_timer.tic()
            self.read_timer.tic()

            for itr, (inputs, targets, metas) in enumerate(self.batch_generator):
                self.set_lr(epoch, itr)
                self.read_timer.toc()
                self.gpu_timer.tic()

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
                self.optimizer.zero_grad()
                sdf_results, hand_pose_results, obj_pose_results, inter_results, processed_gt = self.model(inputs, targets, metas, 'train')

                # loss
                loss_dict = {}
                loss_dict['hand_sdf'] = cfg.TRAIN.hand_sdf_weight * self.loss['hand_sdf'](sdf_results['hand'] * inter_results['mask_hand'], processed_gt['sdf_gt_hand'] * inter_results['mask_hand']) / inter_results['mask_hand'].sum()
                loss_dict['obj_sdf'] = cfg.TRAIN.obj_sdf_weight * self.loss['obj_sdf'](sdf_results['obj'] * inter_results['mask_obj'], processed_gt['sdf_gt_obj'] * inter_results['mask_obj']) / inter_results['mask_obj'].sum()
                loss_dict['volume_joint'] = cfg.TRAIN.volume_weight * self.loss['volume_joint'](obj_pose_results['center'], targets['obj_center_3d'].unsqueeze(1))
                if cfg.MODEL.obj_rot and cfg.TRAIN.corner_weight > 0:
                    loss_dict['obj_corners'] = cfg.TRAIN.corner_weight * self.loss['obj_corners'](obj_pose_results['corners'], targets['obj_corners_3d'])
                if sdf_results['cls_hand'] is not None:
                    if metas['epoch'] >= cfg.TRAIN.sdf_add_epoch:
                        loss_dict['hand_cls'] = cfg.TRAIN.hand_cls_weight * self.loss['hand_cls'](sdf_results['cls_hand'], processed_gt['cls_data'])
                    else:
                        loss_dict['hand_cls'] = 0. * self.loss['hand_cls'](sdf_results['cls_hand'], processed_gt['cls_data'])

                # backward
                all_loss = sum(loss_dict[k].mean() for k in loss_dict)
                all_loss.backward()

                self.optimizer.step()
                torch.cuda.synchronize()

                self.gpu_timer.toc()
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.TRAIN.end_epoch, itr, self.itr_per_epoch),
                    'lr: %g' % (self.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (self.tot_timer.average_time, self.gpu_timer.average_time, self.read_timer.average_time),
                    '%.2fs/epoch' % (self.tot_timer.average_time * self.itr_per_epoch),
                    ]

                # save
                record_dict = {}
                for k, v in loss_dict.items():
                    record_dict[k] = v.detach().mean() * 1000.
                screen += ['%s: %.3f' % ('loss_' + k, v) for k, v in record_dict.items()]

                tb_writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                if itr % 10 == 0:
                    self.logger.info(' '.join(screen))
                    for k, v in record_dict.items():
                        tb_writer.add_scalar('loss_' + k, v, global_steps)
                    tb_writer.add_scalar('lr', self.get_lr(), global_steps)
                    writer_dict['train_global_steps'] = global_steps + 10

                self.tot_timer.toc()
                self.tot_timer.tic()
                self.read_timer.tic()
            
            if (epoch % cfg.model_save_freq == 0 or epoch == cfg.TRAIN.end_epoch - 1):
                self.save_model({
                    'epoch': epoch,
                    'network': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, epoch)
                writer_dict['writer'].close()


class Tester(BaseTester):
    def __init__(self, args, test_epoch):
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name='test_logs.txt')
    
    def _make_model(self, args):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()

    def run(self):
        self.batch_generator, self.itr_per_epoch = get_dataloader(cfg.DATASET.trainset_3d, cfg.DATASET.trainset_3d_split, False, logger=None)
        self._make_model(args)

        with torch.no_grad():
            for itr, (inputs, metas) in enumerate(self.batch_generator): ######## need to re-implement tqdm ########
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
                sdf_feat, hand_pose_results, obj_pose_results = self.model(inputs, targets=None, metas=metas, mode='test')

                # save
                from recon import reconstruct
                from lib.utils.dir_utils import export_pose_results
                export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
                export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)
                reconstruct(cfg, metas['id'], self.model, sdf_feat, inputs, metas, hand_pose_results, obj_pose_results)


def prepare_network(args, load_dir='', is_train=True): 
    from lib.models.model import get_model  
    model = get_model(cfg, is_train)
    if load_dir and (not is_train or args.resume_training):
        checkpoint = load_checkpoint(load_dir=load_dir)
        try:
            model.load_weights(checkpoint['model_state_dict'])
        except:
            model.load_weights(checkpoint)
    else:
        checkpoint = None
    return model, checkpoint
    

def prepare_criterion():
    from lib.core.loss import get_loss
    criterion = get_loss()
    return criterion