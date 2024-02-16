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


class Base(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
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
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        exec(f'from data.{cfg.DATASET.trainset_3d}.dataset import {cfg.DATASET.trainset_3d}')
        trainset3d_db = eval(cfg.DATASET.trainset_3d)('train_' + cfg.DATASET.trainset_3d_split)
        self.trainset_loader = SDFDataset(trainset3d_db, cfg=cfg)
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / cfg.OTHERS.num_gpus / cfg.TRAIN.train_batch_size)
        self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=cfg.TRAIN.train_batch_size * cfg.OTHERS.num_gpus, shuffle=False, num_workers=cfg.OTHERS.num_threads, pin_memory=True, drop_last=True, persistent_workers=False)

    def _make_model(self):
        # prepare network
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


class Tester(Base):
    def __init__(self, test_epoch):
        # self.local_rank = local_rank
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