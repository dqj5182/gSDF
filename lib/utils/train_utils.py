import math
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