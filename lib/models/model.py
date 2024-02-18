import torch
import torch.nn as nn
from lib.core.config import cfg
from lib.models.resnet import ResNetBackbone
from lib.models.unet import UNet
from lib.models.sdf_head import SDFHead
from lib.models.module import FCHead
from lib.models.module import ConvHead
from external.mano.inverse_kinematics import ik_solver_mano
from external.mano.rodrigues_layer import batch_rodrigues
from external.mano.rot6d import compute_rotation_matrix_from_ortho6d
from lib.utils.pose_utils import soft_argmax, decode_volume, decode_volume_abs
from lib.utils.sdf_utils import kinematic_embedding, pixel_align


class HandModel(nn.Module):
    def __init__(self, cfg, backbone, neck, volume_head):
        super(HandModel, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.dim_backbone_feat = 2048 if self.cfg.MODEL.backbone_pose == 'resnet_50' else 512
        self.neck = neck
        self.volume_head = volume_head
        for p in self.parameters():
           p.requires_grad = False
    
    def forward(self, inputs, metas=None):
        input_img = inputs['img']
        backbone_feat = self.backbone(input_img)
        hm_feat = self.neck(backbone_feat)
        hm_pred = self.volume_head(hm_feat)
        hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 21)
        volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
        
        if self.cfg.MODEL.hand_branch:
            hand_pose_results = ik_solver_mano(None, volume_joint_preds[:, :21])
            hand_pose_results['volume_joints'] = volume_joint_preds
        else:
            hand_pose_results = None
        
        return hand_pose_results


class Model(nn.Module):
    def __init__(self, cfg, HandModel, backbone, neck, volume_head, rot_head, hand_sdf_head, obj_sdf_head):
        super(Model, self).__init__()
        self.cfg = cfg
        self.handmodel = HandModel
        self.backbone = backbone
        self.neck = neck
        self.volume_head = volume_head
        self.rot_head = rot_head
        self.dim_backbone_feat = 2048 if self.cfg.MODEL.backbone_shape == 'resnet_50' else 512
        self.hand_sdf_head = hand_sdf_head
        self.obj_sdf_head = obj_sdf_head

        self.backbone_2_sdf = UNet(self.dim_backbone_feat, 256, 1)
        if self.cfg.MODEL.with_add_feats:
            self.sdf_encoder = nn.Linear(260, self.cfg.MODEL.sdf_latent)
        else:
            self.sdf_encoder = nn.Linear(256, self.cfg.MODEL.sdf_latent)
    
    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            if self.cfg.MODEL.hand_branch and self.cfg.MODEL.obj_branch:
                sdf_data = torch.cat([targets['hand_sdf'], targets['obj_sdf']], 1)
                cls_data = torch.cat([targets['hand_labels'], targets['obj_labels']], 1)
                if metas['epoch'] < self.cfg.TRAIN.sdf_add_epoch:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.zeros(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1)
                else:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1)
            elif self.cfg.MODEL.hand_branch:
                sdf_data = targets['hand_sdf']
                cls_data = targets['hand_labels']
                mask_hand = torch.ones(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1).cuda()
            elif self.cfg.MODEL.obj_branch:
                sdf_data = targets['obj_sdf']
                cls_data = targets['obj_labels']
                mask_hand = torch.ones(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points).unsqueeze(1).cuda()

            inter_results = {}
            inter_results['mask_hand'] = mask_hand
            inter_results['mask_obj'] = mask_obj


            sdf_data = sdf_data.reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points, -1)
            cls_data = cls_data.to(torch.long).reshape(self.cfg.TRAIN.train_batch_size * self.cfg.TRAIN.num_sample_points)
            xyz_points = sdf_data[:, 0:-2]
            sdf_gt_hand = sdf_data[:, -2].unsqueeze(1)
            sdf_gt_obj = sdf_data[:, -1].unsqueeze(1)
            if self.cfg.MODEL.hand_branch:
                sdf_gt_hand = torch.clamp(sdf_gt_hand, -self.cfg.TRAIN.clamp_dist, self.cfg.TRAIN.clamp_dist)
            if self.cfg.MODEL.obj_branch:
                sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.cfg.TRAIN.clamp_dist, self.cfg.TRAIN.clamp_dist)

            processed_gt = {}
            processed_gt['sdf_gt_hand'] = sdf_gt_hand
            processed_gt['sdf_gt_obj'] = sdf_gt_obj
            processed_gt['cls_data'] = cls_data


            with torch.no_grad():
                hand_pose_results = self.handmodel(inputs, metas)

            # go through backbone
            backbone_feat = self.backbone(input_img)

            # go through deconvolution
            if self.cfg.MODEL.obj_branch:
                obj_pose_results = {}
                hm_feat = self.neck(backbone_feat)
                hm_pred = self.volume_head(hm_feat)
                hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)
                volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])

                obj_transform = torch.zeros((input_img.shape[0], 4, 4)).to(input_img.device)
                obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d']
                obj_transform[:, 3, 3] = 1
                if self.rot_head is not None:
                    rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                    if cfg.rot_style == 'axisang':
                        obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                    elif cfg.rot_style == '6d':
                        obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                    obj_transform[:, :3, :3] = obj_rot_matrix
                    obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                    obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                else:
                    obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                obj_pose_results['global_trans'] = obj_transform
                obj_pose_results['center'] = volume_joint_preds
                obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                obj_pose_results['joints'] = hand_pose_results['volume_joints'] - metas['hand_center_3d'].unsqueeze(1)
            else:
                obj_pose_results = None

            # generate features for the sdf head
            sdf_feat = self.backbone_2_sdf(backbone_feat)
            sdf_feat, _ = pixel_align(self.cfg, xyz_points, self.cfg.TRAIN.num_sample_points, sdf_feat, metas['hand_center_3d'], metas['cam_intr'])
            sdf_feat = self.sdf_encoder(sdf_feat)

            if self.hand_sdf_head is not None:
                if self.cfg.MODEL.hand_encode_style == 'kine':
                    hand_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.TRAIN.num_sample_points, hand_pose_results, 'hand')
                    hand_points = hand_points.reshape((-1, self.cfg.MODEL.hand_point_latent))
                else:
                    hand_points = xyz_points.reshape((-1, self.cfg.MODEL.hand_point_latent))
                hand_sdf_decoder_inputs = torch.cat([sdf_feat, hand_points], dim=1)
                sdf_hand, cls_hand = self.hand_sdf_head(hand_sdf_decoder_inputs)
                sdf_hand = torch.clamp(sdf_hand, min=-self.cfg.TRAIN.clamp_dist, max=self.cfg.TRAIN.clamp_dist)
            else:
                sdf_hand = None
                cls_hand = None
        
            if self.obj_sdf_head is not None:
                if self.cfg.MODEL.obj_encode_style == 'kine':
                    obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.TRAIN.num_sample_points, obj_pose_results, 'obj')
                    obj_points = obj_points.reshape((-1, self.cfg.MODEL.obj_point_latent))
                else:
                    obj_points = xyz_points.reshape((-1, self.cfg.MODEL.obj_point_latent))
                obj_sdf_decoder_inputs = torch.cat([sdf_feat, obj_points], dim=1)
                sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
                sdf_obj = torch.clamp(sdf_obj, min=-self.cfg.TRAIN.clamp_dist, max=self.cfg.TRAIN.clamp_dist)
            else:
                sdf_obj = None

            sdf_results = {}
            sdf_results['hand'] = sdf_hand
            sdf_results['obj'] = sdf_obj
            sdf_results['cls_hand'] = cls_hand

            return sdf_results, hand_pose_results, obj_pose_results, inter_results, processed_gt
        else:
            with torch.no_grad():
                input_img = inputs['img']
                hand_pose_results = self.handmodel(inputs, metas)
                backbone_feat = self.backbone(input_img)

                if self.cfg.MODEL.obj_branch:
                    obj_pose_results = {}
                    hm_feat = self.neck(backbone_feat)
                    hm_pred = self.volume_head(hm_feat)
                    hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)

                    volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
                    obj_transform = torch.zeros((input_img.shape[0], 4, 4)).to(input_img.device)
                    obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d']
                    obj_transform[:, 3, 3] = 1

                    if self.rot_head is not None:
                        rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                        if cfg.rot_style == 'axisang':
                            obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                        elif cfg.rot_style == '6d':
                            obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                        obj_transform[:, :3, :3] = obj_rot_matrix
                        obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                        obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                    else:
                        obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                    obj_pose_results['global_trans'] = obj_transform
                    obj_pose_results['center'] = volume_joint_preds
                    obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                    obj_pose_results['joints'] = hand_pose_results['volume_joints'] - metas['hand_center_3d'].unsqueeze(1)
                else:
                    obj_pose_results = None

            return backbone_feat, hand_pose_results, obj_pose_results


def get_model(cfg, is_train):
    num_pose_resnet_layers = int(cfg.MODEL.backbone_pose.split('_')[-1])
    num_shape_resnet_layers = int(cfg.MODEL.backbone_shape.split('_')[-1])

    backbone_pose = ResNetBackbone(num_pose_resnet_layers)
    backbone_shape = ResNetBackbone(num_shape_resnet_layers)

    backbone_pose.init_weights()
    backbone_shape.init_weights()

    neck_inplanes = 2048 if num_pose_resnet_layers == 50 else 512
    neck = UNet(neck_inplanes, 256, 3)
    if cfg.MODEL.hand_branch:
        volume_head_hand = ConvHead([256, 21 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
    posenet = HandModel(cfg, backbone_pose, neck, volume_head_hand)

    if cfg.MODEL.obj_branch:
        neck_shape = UNet(neck_inplanes, 256, 3)
        volume_head_obj = ConvHead([256, 1 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
        if cfg.MODEL.obj_rot:
            if cfg.rot_style == 'axisang':
                rot_head_obj = FCHead(out_dim=3)
            elif cfg.rot_style == '6d':
                rot_head_obj = FCHead(out_dim=6)
        else:
            rot_head_obj = None
    else:
        neck_shape = None
        volume_head_obj = None
        rot_head_obj = None

    if cfg.MODEL.hand_branch:
        hand_sdf_head = SDFHead(cfg.MODEL.sdf_latent, cfg.MODEL.hand_point_latent, cfg.MODEL.sdf_head['dims'], cfg.MODEL.sdf_head['dropout'], cfg.MODEL.sdf_head['dropout_prob'], cfg.MODEL.sdf_head['norm_layers'], cfg.MODEL.sdf_head['latent_in'], cfg.MODEL.hand_cls, cfg.MODEL.sdf_head['num_class'])
    else:
        hand_sdf_head = None
    
    if cfg.MODEL.obj_branch:
        obj_sdf_head = SDFHead(cfg.MODEL.sdf_latent, cfg.MODEL.obj_point_latent, cfg.MODEL.sdf_head['dims'], cfg.MODEL.sdf_head['dropout'], cfg.MODEL.sdf_head['dropout_prob'], cfg.MODEL.sdf_head['norm_layers'], cfg.MODEL.sdf_head['latent_in'], False, cfg.MODEL.sdf_head['num_class'])
    else:
        obj_sdf_head = None
    
    ho_model = Model(cfg, posenet, backbone_shape, neck_shape, volume_head_obj, rot_head_obj, hand_sdf_head, obj_sdf_head)

    return ho_model