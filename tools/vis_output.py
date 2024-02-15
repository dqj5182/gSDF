import argparse
import yaml
import json
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import shutil
from multiprocessing import Process, Queue
import trimesh
from tqdm import tqdm
import cv2
import pyrender
import _init_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str)
    parser.add_argument('--id', '-i', default=None, type=str)
    parser.add_argument('--dest_dir', '-d', default=None, type=str)
    parser.add_argument('--mode', '-m', default='overlay_pred', type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # testset = args.dir.strip('/').split('/')[-1].split('_')[1]
    if 'obman' in args.dir:
        testset = 'obman'
    if 'dexycb' in args.dir:
        testset = 'dexycb'
    exec(f'from datasets.{testset}.{testset} import {testset}')

    with open(os.path.join(args.dir, '../exp.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    if testset == 'obman':
        data_root = '../datasets/obman/data/'
        data_json = '../datasets/obman/obman_test.json'
        gt_mesh_hand_source = '../datasets/obman/data/test/mesh_hand/'
        gt_mesh_obj_source = '../datasets/obman/data/test/mesh_obj/'
    elif testset == 'dexycb':
        data_root = '../datasets/dexycb/data/'
        data_json = '../datasets/dexycb/dexycb_test_s0.json'
        from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS
        gt_mesh_hand_source = '../datasets/dexycb/data/mesh_data/mesh_hand/'
        gt_mesh_obj_source = '../datasets/dexycb/data/mesh_data/mesh_obj/'

    output_vis_dir = os.path.join(args.dir, 'vis')
    os.makedirs(output_vis_dir, exist_ok=True)

    with open(data_json, 'r') as f:
        meta_data = json.load(f)

    idx_list = []
    fileanme_list = []
    if args.dest_dir is not None:
        chosen_lists = [filename for filename in os.listdir(args.dest_dir)]
        for i in range(len(meta_data['images'])):
            for chosen_filename in chosen_lists:
                if chosen_filename in str(meta_data['images'][i]['file_name']):
                    idx_list.append(i)
                    fileanme_list.append(meta_data['images'][i]['file_name'])
    else:
        for i in range(len(meta_data['images'])):
            # if str(args.id) in str(meta_data['images'][i]['file_name']):
            idx_list.append(i)
            fileanme_list.append(meta_data['images'][i]['file_name'])

    for idx, sample_id in zip(idx_list, fileanme_list):
        if testset == 'obman':
            img_path = os.path.join(data_root, 'test', 'rgb', sample_id + '.jpg')
            fx = 480.
            fy = 480
            cx = 128.
            cy = 128.
        elif testset == 'dexycb':
            subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
            video_id = '_'.join(sample_id.split('_')[1:3])
            cam_id = sample_id.split('_')[-2]
            frame_id = sample_id.split('_')[-1].rjust(6, '0')
            img_path = os.path.join(data_root, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
            fx = meta_data['annotations'][idx]['fx']
            fy = meta_data['annotations'][idx]['fy']
            cx = meta_data['annotations'][idx]['cx']
            cy = meta_data['annotations'][idx]['cy']

        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_vis_dir, f'{sample_id}.jpg'), img)

        cam_intr = np.zeros((3, 4), dtype=np.float32)
        cam_intr[:3, :3] = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)

        theta_y = -30 / 180 * np.pi
        theta_x = -45 / 180 * np.pi
        rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        camera_rot = rot_x @ rot_y
        rot_center = np.array(meta_data['annotations'][idx]['hand_joints_3d'])[[0], :]

        hand_mesh = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_hand.ply'), process=False)
        hand_mesh.export(os.path.join(output_vis_dir, f'{sample_id}_hand.ply'))
        hand_verts = hand_mesh.vertices
        if 'rot' in args.mode:
            hand_verts = np.dot(camera_rot, (hand_verts - rot_center).transpose(1, 0)).transpose(1, 0)
            hand_verts[:, 2] += rot_center[:, 2]
        hand_verts[:, 1] *= -1
        hand_verts[:, 2] *= -1
        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_mesh.faces)
        hand_mesh.visual.vertex_colors = [78, 179, 211]

        obj_mesh = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_obj.ply'), process=False)
        obj_mesh.export(os.path.join(output_vis_dir, f'{sample_id}_obj.ply'))
        obj_verts = obj_mesh.vertices
        if args.mode == 'rot':
            obj_verts = np.dot(camera_rot, (obj_verts - rot_center).transpose(1, 0)).transpose(1, 0)
            obj_verts[:, 2] += rot_center[:, 2]
        obj_verts[:, 1] *= -1
        obj_verts[:, 2] *= -1
        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh.faces)
        obj_mesh.visual.vertex_colors = [254, 217, 118]

        fuse_mesh = trimesh.util.concatenate(hand_mesh, obj_mesh)
        mesh = pyrender.Mesh.from_trimesh(fuse_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        light = pyrender.DirectionalLight(color=[1.0, 0.0, 0.0], intensity=4.0)
        scene.add(light)
        light = pyrender.DirectionalLight(color=[0.0, 1.0, 0.0], intensity=4.0)
        scene.add(light)
        light = pyrender.DirectionalLight(color=[0.0, 0.0, 1.0], intensity=4.0)
        scene.add(light)
        camera = pyrender.IntrinsicsCamera(fx=cam_intr[0, 0], fy=cam_intr[1, 1], cx=cam_intr[0, 2], cy=cam_intr[1, 2])
        scene.add(camera, pose=np.eye(4))
        if testset == 'obman':
            r = pyrender.OffscreenRenderer(256, 256)
        elif testset == 'dexycb':
            r = pyrender.OffscreenRenderer(640, 480)

        color, depth = r.render(scene)
        if 'rot' in args.mode:
            overlay = color[:, :, ::-1]
        else:
            color = color[:, :, ::-1]
            mask = color[:, :, [0]] > 250
            overlay = img * mask + color * (1 - mask)

        import pdb; pdb.set_trace()
        # cv2.imwrite('debug_img.png', img)
        # cv2.imwrite('debug_mask.png', mask*255)
        # cv2.imwrite('debug_color.png', color)
        # cv2.imwrite('debug_overlay.png', overlay)
        cv2.imwrite(os.path.join(output_vis_dir, f'{sample_id}_{args.mode}.jpg'), overlay)
        r.delete()