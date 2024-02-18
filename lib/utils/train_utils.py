import math
import torch
from torch.utils.data import DataLoader
from lib.core.config import cfg
from lib.datasets.sdf_dataset import SDFDataset

def get_dataloader(dataset_name, dataset_split, is_train, logger=None):
    if is_train is True:
        logger.info("Creating dataset...")
        exec(f'from data.{dataset_name}.dataset import {dataset_name}')
        trainset3d_db = eval(dataset_name)('train_' + dataset_split)
        trainset_loader = SDFDataset(trainset3d_db, cfg=cfg)
        itr_per_epoch = math.ceil(len(trainset_loader) / cfg.OTHERS.num_gpus / cfg.TRAIN.train_batch_size)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.TRAIN.train_batch_size * cfg.OTHERS.num_gpus, shuffle=False, num_workers=cfg.OTHERS.num_threads, pin_memory=True, drop_last=True, persistent_workers=False)
    else:
        exec(f'from data.{dataset_name}.dataset import {dataset_name}')
        testset3d_db = eval(dataset_name)('test_' + dataset_split)
        testset_loader = SDFDataset(testset3d_db, cfg=cfg, mode='test')
        itr_per_epoch = math.ceil(len(testset_loader) / cfg.OTHERS.num_gpus / cfg.TEST.test_batch_size)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.TEST.test_batch_size * cfg.OTHERS.num_gpus, shuffle=False, num_workers=cfg.OTHERS.num_threads, pin_memory=True, drop_last=False, persistent_workers=False)
    return batch_generator, itr_per_epoch


def train_setup(model, checkpoint):    
    optimizer = self.get_optimizer(model)

    ckpt = torch.load(cfg.MODEL.weight_path, map_location=torch.device('cpu'))['network']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    self.model.module.handmodel.load_state_dict(ckpt)
    self.logger.info('Load checkpoint from {}'.format(cfg.MODEL.weight_path))
    self.model.module.handmodel.eval()

    # load_model
    model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
    if len(model_file_list) == 0:
        if os.path.exists(cfg.OTHERS.checkpoint):
            ckpt = torch.load(cfg.OTHERS.checkpoint, map_location=torch.device('cpu'))
            self.model.load_state_dict(ckpt['network'])
            start_epoch = 0
            self.logger.info('Load checkpoint from {}'.format(cfg.OTHERS.checkpoint))
        else:
            start_epoch = 0
            self.logger.info('Start training from scratch')
    else:
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) 
        start_epoch = ckpt['epoch'] + 1
        self.model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        self.logger.info('Continue training and load checkpoint from {}'.format(ckpt_path))

    return optimizer, loss_history, eval_history



def get_optimizer(model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.lr)
    return optimizer