import os
import sys
import argparse
import glob
import pickle
import numpy as np
import cv2
import torch
import multiprocessing as mp

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from torch.utils.data import Dataset, DataLoader


# 프로젝트 유틸 경로 추가
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from nusc_image_projection import (
    to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
)
from tqdm import tqdm 
from matplotlib import pyplot as plt

H, W = 810, 1280  # 투영할 이미지 크기

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar_dir', type=str, required=True, help='레이더 .bin 파일들이 있는 디렉토리')
    parser.add_argument('--image_dir', type=str, required=True, help='카메라 이미지 디렉토리 (투영 결과 미리보기/검증용)')
    parser.add_argument('--output_dir', type=str, required=True, help='투영 결과 저장 디렉토리')
    parser.add_argument('--save_overlay', action='store_true', help='이미지에 오버레이 PNG도 함께 저장')
    parser.add_argument('--save_raw', action='store_true', help='투영 전 원시 포인트도 함께 저장')

    return parser

def setup_cfg(args):
  
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def simple_collate(batch_list):
    assert len(batch_list) == 1
    batch_list = batch_list[0]
    return batch_list 

class Datasetvod(Dataset):
    def __init__(
        self,
        info_path,
        predictor
    ):
        self.sweeps = get_obj(info_path)
        self.predictor = predictor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]

        img_name = info['image']['image_idx'] + '.jpg'
        img_path = os.path.join('./data/vod/training/image_2/', img_name)
        original_image = cv2.imread(img_path)

        if self.predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        
        inputs = {"image": image, "height": height, "width": width}

        return [info, inputs]
    
    def __len__(self):
        return len(self.sweeps)
    

def read_file(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    return points



def render_overlay(image_path, uv, mask, color=(0, 165, 255), radius=2):
    """
    투영된 포인트를 이미지에 찍어서 시각화.
    color는 BGR(주황 예시), radius는 점 크기.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    pts = uv[mask]
    for x, y in pts.astype(int):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, color, -1)
    return img

def save_projection(output_dir, key, proj_dict, raw_dict=None, overlay_img=None, ext='.npz'):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(key))[0]
    base = os.path.join(output_dir, stem)

    if ext == '.npz':
        np.savez_compressed(
            base + '.npz',
            uv=proj_dict['uv'],
            mask=proj_dict['mask'].astype(np.uint8),
            rcs=proj_dict['attr']['rcs'],
            vel_xy=proj_dict['attr']['vel_xy']
        )
    elif ext == '.pkl':
        with open(base + '.pkl', 'wb') as f:
            pickle.dump(proj_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if raw_dict is not None:
        np.savez_compressed(base + '_raw.npz',
                            points_xyz=raw_dict['points_xyz'],
                            vel_xy=raw_dict['vel_xy'],
                            rcs=raw_dict['rcs'])

    if overlay_img is not None:
        cv2.imwrite(base + '_overlay.png', overlay_img)



def process_single_radar(info, predictor, data):
    all_cams_from_lidar = [info['calib']['Tr_velo_to_cam']] # 4*4
    all_cams_intrinsic = [info['calib']['P2'][:3, :3]] # 3*3

    pts_name = info['point_cloud']['lidar_idx']+'.bin'
    pts_path = os.path.join('./data/vod/training/velodyne/', pts_name)
    lidar_points = read_file(pts_path)

    # lidar projection 
    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic), H=H, W=W)
    camera_ids = torch.arange(1, dtype=torch.float32, device='cuda:0').reshape(1, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1) 

    return P

def main():
    mp.set_start_method('spawn', force=True)
    args = get_parser().parse_args()
    setup_logger(name='fvcore')
    logger = setup_logger()
    logger.info('Arguments: ' + str(args))
    cfg = setup_cfg(args)
    
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor
    dataset = Datasetvod('data/vod/kitti_infos_trainval.pkl', predictor)

    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=0,
        collate_fn=simple_collate, 
        pin_memory=False, 
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue
        info=data[0]
        output_name = info['image']['image_idx']+'.jpg'
        save_path = './data/vod/training/projected_radar/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        output_path = os.path.join(save_path, output_name)
        projected_pts = process_single_radar(info, predictor, data)

if __name__ == '__main__':
    main()
