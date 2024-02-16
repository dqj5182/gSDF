import os
import os.path as osp
import sys
import yaml
from yacs.config import CfgNode as CN
from easydict import EasyDict as edict
from loguru import logger
from contextlib import redirect_stdout


cfg = edict() #CN()

cfg.task = 'hsdf_osdf_2net_pa'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.sdf_result_dir = '.'
cfg.cls_sdf_result_dir = '.'
cfg.hand_pose_result_dir = '.'
cfg.obj_pose_result_dir = '.'

## Dataset
cfg.DATASET = edict()
cfg.DATASET.trainset_3d = 'obman'
cfg.DATASET.trainset_3d_split = '87k'
cfg.DATASET.testset = 'obman'
cfg.DATASET.testset_split = '6k'
cfg.DATASET.testset_hand_source = osp.join(cfg.DATASET.testset, 'data/test/mesh_hand')
cfg.DATASET.testset_obj_source = osp.join(cfg.DATASET.testset, 'data/test/mesh_obj')
cfg.DATASET.num_testset_samples = 6285
cfg.DATASET.mesh_resolution = 128
cfg.DATASET.point_batch_size = 2 ** 18
cfg.DATASET.output_part_label = False
cfg.DATASET.vis_part_label = False
cfg.DATASET.chamfer_optim = True

## Model
cfg.MODEL = edict()
cfg.MODEL.backbone_pose = 'resnet_18'
cfg.MODEL.backbone_shape = 'resnet_18'
cfg.MODEL.mano_pca_latent = 15
cfg.MODEL.sdf_latent = 256
cfg.MODEL.hand_point_latent = 3
cfg.MODEL.obj_point_latent = 3
cfg.MODEL.hand_encode_style = 'nerf'
cfg.MODEL.obj_encode_style = 'nerf'
cfg.MODEL.rot_style = '6d'
cfg.MODEL.hand_branch = True
cfg.MODEL.obj_branch = True
cfg.MODEL.hand_cls = False
cfg.MODEL.obj_rot = False
cfg.MODEL.with_add_feats = True
cfg.MODEL.ckpt = '.'

cfg.MODEL.sdf_head = CN()
cfg.MODEL.sdf_head.layers = 5
cfg.MODEL.sdf_head.dims = [512 for i in range(cfg.MODEL.sdf_head.layers - 1)]
cfg.MODEL.sdf_head.dropout = [i for i in range(cfg.MODEL.sdf_head.layers - 1)]
cfg.MODEL.sdf_head.norm_layers = [i for i in range(cfg.MODEL.sdf_head.layers - 1)]
cfg.MODEL.sdf_head.dropout_prob = 0.2
cfg.MODEL.sdf_head.latent_in = [(cfg.MODEL.sdf_head.layers - 1) // 2]
cfg.MODEL.sdf_head.num_class = 6

## Training
cfg.TRAIN = edict()
cfg.TRAIN.image_size = (256, 256)
cfg.TRAIN.heatmap_size = (64, 64, 64)
cfg.TRAIN.depth_dim = 0.28
cfg.TRAIN.warm_up_epoch = 0
cfg.TRAIN.lr_dec_epoch = [600, 1200]
cfg.TRAIN.end_epoch = 1600
cfg.TRAIN.sdf_add_epoch = 1201
cfg.TRAIN.lr = 1e-4
cfg.TRAIN.lr_dec_style = 'step'
cfg.TRAIN.lr_dec_factor = 0.5
cfg.TRAIN.train_batch_size = 64
cfg.TRAIN.num_sample_points = 2000
cfg.TRAIN.clamp_dist = 0.05
cfg.TRAIN.recon_scale = 6.5
cfg.TRAIN.hand_sdf_weight = 0.5
cfg.TRAIN.obj_sdf_weight = 0.5
cfg.TRAIN.hand_cls_weight = 0.05
cfg.TRAIN.volume_weight = 0.5
cfg.TRAIN.corner_weight = 0.5
cfg.TRAIN.use_inria_aug = False
cfg.TRAIN.norm_factor = 0.02505871

## Testing
cfg.TEST = edict()
cfg.TEST.test_batch_size = 1
cfg.TEST.test_with_gt = False

## Others
cfg.OTHERS = edict()
cfg.OTHERS.num_threads = 6
cfg.OTHERS.gpu_ids = (0, 1, 2, 3)
cfg.OTHERS.num_gpus = 4
cfg.OTHERS.checkpoint = 'model.pth.tar'
cfg.OTHERS.model_save_freq = 100

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            pass

def update_config(config_file, args, mode='train'):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, yaml.SafeLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

    # cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    cfg.OTHERS.gpu_ids = args.gpu_ids
    cfg.OTHERS.num_gpus = len(cfg.OTHERS.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.OTHERS.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.OTHERS.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.DATASET.trainset_3d + cfg.DATASET.trainset_3d_split, cfg.MODEL.backbone_pose.replace('_', ''), cfg.MODEL.backbone_shape.replace('_', ''), 'h' + str(int(cfg.MODEL.hand_branch)), 'o' + str(int(cfg.MODEL.obj_branch)), 'sdf' + str(cfg.MODEL.sdf_head.layers), 'cls' + str(int(cfg.MODEL.hand_cls)), 'rot' + str(int(cfg.MODEL.obj_rot)), 'hand_' + cfg.MODEL.hand_encode_style + '_' + str(cfg.MODEL.hand_point_latent), 'obj_' + cfg.MODEL.obj_encode_style + '_' + str(cfg.MODEL.obj_point_latent), 'np' + str(cfg.TRAIN.num_sample_points), 'adf' + str(int(cfg.MODEL.with_add_feats)), 'e' + str(cfg.TRAIN.end_epoch), 'ae' + str(cfg.TRAIN.sdf_add_epoch), 'scale' + str(cfg.TRAIN.recon_scale), 'b' + str(cfg.OTHERS.num_gpus * cfg.TRAIN.train_batch_size), 'hsw' + str(cfg.TRAIN.hand_sdf_weight), 'osw' + str(cfg.TRAIN.obj_sdf_weight), 'hcw' + str(cfg.TRAIN.hand_cls_weight), 'vw' + str(cfg.TRAIN.volume_weight)]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.DATASET.testset, 'gt', str(int(cfg.TEST.test_with_gt))]))
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = osp.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = osp.join(cfg.result_dir, 'obj_pose')

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        # cfg.freeze()
        # with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
        #     with redirect_stdout(f): print(cfg.dump())
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    else:
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.DATASET.testset, 'gt', str(int(cfg.TEST.test_with_gt))]))
        cfg.log_dir = osp.join(cfg.output_dir, 'test_log')
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        # cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from lib.utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.DATASET.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.DATASET.testset))