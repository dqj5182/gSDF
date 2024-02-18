import os
import os.path as osp
import math
import glob
import abc
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from lib.utils.timer import Timer
from lib.core.config import cfg
from model import get_model
from lib.datasets.sdf_dataset import SDFDataset


class BaseTrainer:
    def __init__(self, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))

        self.loss = prepare_criterion()


class BaseTester:
    def __init__(self, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))


class Trainer(BaseTrainer):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        if len(model_file_list) == 0:
            if os.path.exists(cfg.OTHERS.checkpoint):
                ckpt = torch.load(cfg.OTHERS.checkpoint, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['network'])
                start_epoch = 0
                self.logger.info('Load checkpoint from {}'.format(cfg.OTHERS.checkpoint))
                return start_epoch, model, optimizer
            else:
                start_epoch = 0
                self.logger.info('Start training from scratch')
                return start_epoch, model, optimizer
        else:
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) 
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['network'])
            optimizer.load_state_dict(ckpt['optimizer'])
            self.logger.info('Continue training and load checkpoint from {}'.format(ckpt_path))
            return start_epoch, model, optimizer

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
    
    def _make_batch_generator(self):
        self.logger.info("Creating dataset...")
        exec(f'from data.{cfg.DATASET.trainset_3d}.dataset import {cfg.DATASET.trainset_3d}')
        trainset3d_db = eval(cfg.DATASET.trainset_3d)('train_' + cfg.DATASET.trainset_3d_split)
        self.trainset_loader = SDFDataset(trainset3d_db, cfg=cfg)
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / cfg.OTHERS.num_gpus / cfg.TRAIN.train_batch_size)
        self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=cfg.TRAIN.train_batch_size * cfg.OTHERS.num_gpus, shuffle=False, num_workers=cfg.OTHERS.num_threads, pin_memory=True, drop_last=True, persistent_workers=False)

    def _make_model(self):
        self.logger.info("Creating graph and optimizer...")
        model = get_model(cfg, True)
        model = model.cuda()
        model = nn.DataParallel(model)
        optimizer = self.get_optimizer(model)
        model.train()

        ckpt = torch.load(cfg.MODEL.weight_path, map_location=torch.device('cpu'))['network']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.module.pose_model.load_state_dict(ckpt)
        self.logger.info('Load checkpoint from {}'.format(cfg.MODEL.weight_path))
        model.module.pose_model.eval()

        start_epoch, model, optimizer = self.load_model(model, optimizer)
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def run(self, writer_dict):
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
                for k, v in loss.items():
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
    def __init__(self, test_epoch):
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        exec(f'from data.{cfg.DATASET.testset}.dataset import {cfg.DATASET.testset}')
        testset3d_db = eval(cfg.DATASET.testset)('test_' + cfg.DATASET.testset_split)

        self.testset_loader = SDFDataset(testset3d_db, cfg=cfg, mode='test')
        self.itr_per_epoch = math.ceil(len(self.testset_loader) / cfg.OTHERS.num_gpus / cfg.TEST.test_batch_size)
        self.batch_generator = DataLoader(dataset=self.testset_loader, batch_size=cfg.TEST.test_batch_size * cfg.OTHERS.num_gpus, shuffle=False, num_workers=cfg.OTHERS.num_threads, pin_memory=True, drop_last=False, persistent_workers=False)
    
    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(cfg, is_train=False)
        model = model.cuda()
        model = nn.DataParallel(model)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model


def prepare_network(args, load_dir='', is_train=True): 
    from lib.models.model import get_model  
    model = get_model(is_train)
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