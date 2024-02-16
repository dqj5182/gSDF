import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.mano import MANO
from utils.fitting import ScaleTranslationLoss, FittingMonitor
from utils.optimizers import optim_factory
from utils.camera import PerspectiveCamera
from config import cfg
import math


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.hand_regHead = hand_regHead()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regHead()
    
    def forward(self, feats, gt_mano_params=None):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)
        mano_encoding = self.hand_Encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_regHead(mano_encoding, gt_mano_params)

        return pred_mano_results, gt_mano_results, preds_joints_img


class Transformer(nn.Module):
    def __init__(self, inp_res=32, dim=256, depth=2, num_heads=4, mlp_ratio=4., injection=True):
        super().__init__()

        self.injection=injection
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, injection=injection))

        if self.injection:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1, padding=0),
            )

    def forward(self, query, key):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key)
        
        if self.injection:
            output = torch.cat([key, output], dim=1)
            output = self.conv1(output) + self.conv2(output)

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, query, key, value, query2, key2, use_sigmoid):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
            
        if use_sigmoid:
            query2 = query2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            key2 = key2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale
            attn2 = torch.sum(attn2, dim=-1)
            attn2 = self.sigmoid(attn2)
            attn = attn * attn2.unsqueeze(3) 
        
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, injection=True):
        super().__init__()

        self.injection = injection

        self.channels = dim

        self.encode_value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        if self.injection:
            self.encode_query2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
            self.encode_key2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.q_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))
        self.k_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, h, w = query.shape
        query_embed = repeat(self.q_embedding, '() n c d -> b n c d', b = b)
        key_embed = repeat(self.k_embedding, '() n c d -> b n c d', b = b)

        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q_embed).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k_embed).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        
        query = query.view(b, self.channels, -1).permute(0, 2, 1)

        if self.injection:
            q2 = self.encode_query2(q_embed).view(b, self.channels, -1)
            q2 = q2.permute(0, 2, 1)

            k2 = self.encode_key2(k_embed).view(b, self.channels, -1)
            k2 = k2.permute(0, 2, 1)

            query = self.attn(query=q, key=k, value=v,query2 = q2, key2 = k2, use_sigmoid=True)
        else:
            q2 = None
            k2 = None

            query = query + self.attn(query=q, key=k, value=v, query2 = q2, key2 = k2, use_sigmoid=False)
 
        query = query + self.mlp(self.norm2(query))
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, h, w)

        return query


class FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN, self).__init__()
        self.in_planes = 64

        resnet = resnet50(pretrained=pretrained)

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.leakyrelu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Attention Module
        self.attention_module = SpatialGate()

        self.pool = nn.AvgPool2d(2, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        #p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        # Attention
        p2 = self.pool(p2)
        primary_feats, secondary_feats = self.attention_module(p2)
        
        return primary_feats, secondary_feats


class HandOccNet(nn.Module):
    def __init__(self, backbone, FIT, SET, regressor):
        super(HandOccNet, self).__init__()
        self.backbone = backbone
        self.FIT = FIT
        self.SET = SET
        self.regressor = regressor
        
        self.fitting_loss = ScaleTranslationLoss(list(range(0, 21))) # fitting joint indices

    
    def forward(self, inputs, targets, meta_info, mode):
        p_feats, s_feats = self.backbone(inputs['img']) # primary, secondary feats
        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)

        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)
       
        if mode == 'train':
            # loss functions
            loss = {}
            loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
            loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
            loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])
            return loss

        else:
            # test output
            out = {}
            out['joints_coord_img'] = preds_joints_img[0]
            out['mano_pose'] = pred_mano_results['mano_pose_aa']
            out['mano_shape'] = pred_mano_results['mano_shape']
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            out['manojoints2cam'] = pred_mano_results['manojoints2cam'] 
            out['mano_pose_aa'] = pred_mano_results['mano_pose_aa']

            return out

    def get_mesh_scale_trans(self, pred_joint_img, pred_joint_cam, init_scale=1., init_depth=1., camera=None, depth_map=None):
        """
        pred_joint_img: (batch_size, 21, 2)
        pred_joint_cam: (batch_size, 21, 3)
        """
        if camera is None:
            camera = PerspectiveCamera()

        dtype, device = pred_joint_cam.dtype, pred_joint_cam.device
        hand_scale = torch.tensor([init_scale / 1.0], dtype=dtype, device=device, requires_grad=False)
        hand_translation = torch.tensor([0, 0, init_depth], dtype=dtype, device=device, requires_grad=True)
        if depth_map is not None:
            tensor_depth = torch.tensor(depth_map, device=device, dtype=dtype)[
                None, None, :, :]
            grid = pred_joint_img.clone()
            grid[:, :, 0] /= tensor_depth.shape[-1]
            grid[:, :, 1] /= tensor_depth.shape[-2]
            grid = 2 * grid - 1
            joints_depth = torch.nn.functional.grid_sample(
                tensor_depth, grid[:, None, :, :])  # (1, 1, 1, 21)
            joints_depth = joints_depth.reshape(1, 21, 1)
            hand_translation = torch.tensor(
                [0, 0, joints_depth[0, cfg.fitting_joint_idxs, 0].mean() / 1000.], device=device, requires_grad=True)

        # intended only for demo mesh rendering
        batch_size = 1
        self.fitting_loss.trans_estimation = hand_translation.clone()

        params = []
        params.append(hand_translation)
        params.append(hand_scale)
        optimizer, create_graph = optim_factory.create_optimizer(
            params, optim_type='lbfgsls', lr=1.0e-1)

        # optimization
        print("[Fitting]: fitting the hand scale and translation...")
        with FittingMonitor(batch_size=batch_size) as monitor:
            fit_camera = monitor.create_fitting_closure(
                optimizer, camera, pred_joint_cam, pred_joint_img, hand_translation, hand_scale, self.fitting_loss, create_graph=create_graph)

            loss_val = monitor.run_fitting(
                optimizer, fit_camera, params)


        print(f"[Fitting]: fitting finished with loss of {loss_val}")
        print(f"Scale: {hand_scale.detach().cpu().numpy()}, Translation: {hand_translation.detach().cpu().numpy()}")
        return hand_scale, hand_translation
    
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True) # feature injecting transformer
    SET = Transformer(injection=False) # self enhancing transformer
    regressor = Regressor()
    
    if mode == 'train':
        FIT.apply(init_weights)
        SET.apply(init_weights)
        regressor.apply(init_weights)
        
    model = HandOccNet(backbone, FIT, SET, regressor)
    
    return model