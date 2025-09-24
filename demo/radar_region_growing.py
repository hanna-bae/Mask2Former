import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood
from tqdm import tqdm
import matplotlib.colors as mcolors
from multiprocessing import Pool, Manager
from matplotlib.colors import Normalize
import cv2
import logging
import argparse 
import glob 
import torch
import multiprocessing as mp 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from nusc_image_projection import projectionV2, to_tensor, to_batch_tensor, get_obj
from calibration_kitti import Calibration
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

H, W = 1216, 1936

# # calibration matrix for test 
# def get_calib_from_file(calib_file):
#     with open(calib_file) as f:
#         lines = f.readlines()

#     obj = lines[2].strip().split(' ')[1:]
#     P2 = np.array(obj, dtype=np.float32)
#     obj = lines[3].strip().split(' ')[1:]
#     P3 = np.array(obj, dtype=np.float32)
#     obj = lines[4].strip().split(' ')[1:]
#     R0 = np.array(obj, dtype=np.float32)
#     obj = lines[5].strip().split(' ')[1:]
#     Tr_velo_to_cam = np.array(obj, dtype=np.float32)

#     return {'P2': P2.reshape(3, 4),
#             'P3': P3.reshape(3, 4),
#             'R0': R0.reshape(3, 3),
#             'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

# class Calibration(object):
#     def __init__(self, calib_file):
#         if not isinstance(calib_file, dict):
#             calib = get_calib_from_file(calib_file)
#         else:
#             calib = calib_file

#         self.P2 = calib['P2']  # 3 x 4
#         self.R0 = calib['R0']  # 3 x 3
#         self.V2C = calib['Tr_velo2cam']  # 3 x 4

#         # Camera intrinsics and extrinsics
#         self.cu = self.P2[0, 2]
#         self.cv = self.P2[1, 2]
#         self.fu = self.P2[0, 0]
#         self.fv = self.P2[1, 1]
#         self.tx = self.P2[0, 3] / (-self.fu)
#         self.ty = self.P2[1, 3] / (-self.fv)

#     def cart_to_hom(self, pts):
#         """
#         :param pts: (N, 3 or 2)
#         :return pts_hom: (N, 4 or 3)
#         """
#         pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
#         return pts_hom

#     def rect_to_lidar(self, pts_rect):
#         """
#         :param pts_lidar: (N, 3)
#         :return pts_rect: (N, 3)
#         """
#         pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
#         R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
#         R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
#         R0_ext[3, 3] = 1
#         V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
#         V2C_ext[3, 3] = 1

#         pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
#         return pts_lidar[:, 0:3]

#     def lidar_to_rect(self, pts_lidar):
#         """
#         :param pts_lidar: (N, 3)
#         :return pts_rect: (N, 3)
#         """
#         pts_lidar_hom = self.cart_to_hom(pts_lidar)
#         pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
#         # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
#         return pts_rect

#     def rect_to_img(self, pts_rect):
#         """
#         :param pts_rect: (N, 3)
#         :return pts_img: (N, 2)
#         """
#         pts_rect_hom = self.cart_to_hom(pts_rect)
#         pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
#         pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
#         pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
#         return pts_img, pts_rect_depth

#     def lidar_to_img(self, pts_lidar):
#         """
#         :param pts_lidar: (N, 3)
#         :return pts_img: (N, 2)
#         """
#         pts_rect = self.lidar_to_rect(pts_lidar)
#         pts_img, pts_depth = self.rect_to_img(pts_rect)
#         return pts_img, pts_depth

#     def img_to_rect(self, u, v, depth_rect):
#         """
#         :param u: (N)
#         :param v: (N)
#         :param depth_rect: (N)
#         :return:
#         """
#         x = ((u - self.cu) * depth_rect) / self.fu + self.tx
#         y = ((v - self.cv) * depth_rect) / self.fv + self.ty
#         pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
#         return pts_rect

#     def corners3d_to_img_boxes(self, corners3d):
#         """
#         :param corners3d: (N, 8, 3) corners in rect coordinate
#         :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
#         :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
#         """
#         sample_num = corners3d.shape[0]
#         corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

#         img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

#         x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
#         x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
#         x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

#         boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
#         boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

#         return boxes, boxes_corner

LOGGER = logging.getLogger(__name__)
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pts-save-path",
        required=False,
        help="path to save path of hybrid points",
    )
    parser.add_argument(
        "--config-file",
        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS",
                "./ckpt/model_final_dfa862.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser

class DatasetVoD(Dataset):
    def __init__(self, info_path, predictor):
        self.sweeps = get_obj(info_path)
        self.predictor = predictor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]
        img_name = info['image']['image_idx'] +'.jpg'
        img_path = os.path.join('./data/vod/radar_5frames/training/image_2/', img_name)
        original_image = cv2.imread(img_path)

        if self.predictor.input_format == 'RGB':
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        return [info, inputs]
    
    def __len__(self):
        return len(self.sweeps)
    
def load_image(path, normalize=True):
    image = Image.open(path).convert('RGB')
    image = np.asarray(image, np.float32)
    if normalize:
        image = image / 255.0
    return image

def load_depth(path, multiplier=256.0):
    z = np.array(Image.open(path), dtype=np.float32)
    z = z / multiplier
    z[z <= 0] = 0.0
    return z

def save_depth(z, path, multiplier=256.0):
    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def radar_to_image_depth(info, raw_pts, image_shape):
    all_cams_from_lidar = [info['calib']['Tr_velo_to_cam']]
    all_cams_intrinsic = [info['calib']['P2'][:3, :3]]
    img_h, img_w = image_shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projected_pts = projectionV2(to_tensor(raw_pts), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic), img_h, img_w)
    # camera_ids = torch.arange(1, dtype=torch.float32, device='cuda:0').reshape(1, 1, 1).repeat(1, projected_pts.shape[1], 1)
    # projected_pts = torch.cat([projected_pts, camera_ids], dim=-1)
    # print(projected_pts.shape)
    # print(projected_pts)
    # print(projected_pts.reshape(-1,4))
    projected_pts = projected_pts.reshape(-1, 4)
    pts_2d = projected_pts[:, :2]
    pts_depth = projected_pts[:, 2]
    # print(pts_2d)
    # print(pts_depth)
    # print(pts_depth.shape)
    xs, ys = pts_2d[:, 0].round().to(torch.int32), pts_2d[:, 1].round().to(torch.int32)
    valid = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h) & (pts_depth > 0)
    xs = xs[valid].cpu().numpy()
    ys = ys[valid].cpu().numpy()
    zs = pts_depth[valid].cpu().numpy()
    # 최근접 깊이 유지: np.minimum.at
    depth_img = np.zeros((img_h, img_w), dtype=np.float32)
    # 깊이가 0이면 “비어있음”이므로, 먼저 아주 큰 값으로 채워놓고 최소 적용 후 다시 0으로 되돌리는 방식
    depth_img[:] = np.inf
    np.minimum.at(depth_img, (ys, xs), zs)
    depth_img[~np.isfinite(depth_img)] = 0.0
    return depth_img

def region_growing_skimage(mono_depth, seed_point, threshold=0.5):
    # print(mono_depth.shape)
    # print(seed_point)
    mono_depth = mono_depth.mean(axis=2)
    seed_y, seed_x = seed_point
    region_mask = flood(mono_depth, (seed_y, seed_x), connectivity=2, tolerance=threshold)
    return region_mask

def crop_region_mask(region_mask, center, patch_size, image_shape):
    H, W = image_shape
    h, w = patch_size
    cy, cx = center
    sy, ey = max(0, cy - h//2), min(H, cy + h//2)
    sx, ex = max(0, cx - w//2), min(W, cx + w//2)
    cropped_region_mask = np.zeros_like(region_mask, dtype=bool)
    cropped_region_mask[sy:ey, sx:ex] = region_mask[sy:ey, sx:ex]
    return cropped_region_mask

def generate_virtual_points_from_depth_map(depth_map, calib_info, radar_point_attrs, num_points):
    y_coords, x_coords = np.where(depth_map > 0)
    if y_coords.size == 0 or num_points == 0:
        return np.empty((0, 8))
    
    num_available_pixels = len(y_coords)
    num_to_sample = min(num_points, num_available_pixels)
    
    sampled_indices = np.random.choice(num_available_pixels, size=num_to_sample, replace=False)
    
    sampled_y = y_coords[sampled_indices]
    sampled_x = x_coords[sampled_indices]
    sampled_depths = depth_map[sampled_y, sampled_x]
    
    # # 투영된 2D 깊이 맵에서 3D 위치 (x,y,z)를 생성
    # points_2d_with_depth = np.vstack((sampled_x, sampled_y, sampled_depths)).T
    # P2 = calib_info['calib']['P2']
    # points_camera = reverse_view_points(points_2d_with_depth, P2) 
    
    # V2C_4x4 = np.concatenate([calib_info['calib']['Tr_velo_to_cam'], np.array([[0., 0., 0., 1.]])], axis=0)
    # C2V_4x4 = np.linalg.inv(V2C_4x4)
    # points_camera_homo = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
    # points_lidar_homo = (C2V_4x4 @ points_camera_homo.T).T
    
    # virtual_points_xyz = points_lidar_homo[:, :3]
    # 1. 2D 이미지 좌표와 깊이 값을 사용하여 Rect 좌표계로 변환
    points_rect = calib_info['calib_obj'].img_to_rect(sampled_x, sampled_y, sampled_depths)

    # 2. Rect 좌표계의 포인트를 Lidar 좌표계로 변환
    virtual_points_xyz = calib_info['calib_obj'].rect_to_lidar(points_rect)

    # 가상 포인트에 원본 속성 전파
    virtual_points_attrs = np.tile(radar_point_attrs, (num_to_sample, 1))

    # 위치 + 속성 결합
    virtual_points_with_attrs = np.hstack((virtual_points_xyz, virtual_points_attrs))

    return virtual_points_with_attrs

def visualize_points_on_image(image, points, calib, filename='visualization.jpg'):
    """
    Visualize points on image
    Args:
        image: np.ndarray, origin image (H, W, 3).
        points: np.ndarray, shape (N, 8), [x, y, z, rcs, v, v', t, flag].
        calib: calibration object
        filename: filenames of saving file
    """
    plt.figure(figsize=(15, 8))
    plt.imshow(image)

    # 1. Separate real and virtual points
    flag = points[:, 7]
    real_pts = points[flag == 0]
    virtual_pts = points[flag == 1]

    # 2. Project real points to image plane
    real_pts_rect = calib.lidar_to_rect(real_pts[:, :3])
    real_pts_img, _ = calib.rect_to_img(real_pts_rect)

    # 3. Project virtual points to image plane
    virtual_pts_rect = calib.lidar_to_rect(virtual_pts[:, :3])
    virtual_pts_img, _ = calib.rect_to_img(virtual_pts_rect)

    # 4. Create and apply a valid mask for REAL points
    real_valid_mask = (real_pts_img[:, 0] >= 0) & (real_pts_img[:, 0] < image.shape[1]) & \
                      (real_pts_img[:, 1] >= 0) & (real_pts_img[:, 1] < image.shape[0]) & \
                      (real_pts_rect[:, 2] > 0)
    
    valid_real_pts_img = real_pts_img[real_valid_mask]

    # 5. Create and apply a valid mask for VIRTUAL points
    virtual_valid_mask = (virtual_pts_img[:, 0] >= 0) & (virtual_pts_img[:, 0] < image.shape[1]) & \
                         (virtual_pts_img[:, 1] >= 0) & (virtual_pts_img[:, 1] < image.shape[0]) & \
                         (virtual_pts_rect[:, 2] > 0)

    valid_virtual_pts_img = virtual_pts_img[virtual_valid_mask]
    
    # 6. Plot the filtered points
    plt.scatter(valid_real_pts_img[:, 0], valid_real_pts_img[:, 1], c='red', s=1, label='Real Points')
    plt.scatter(valid_virtual_pts_img[:, 0], valid_virtual_pts_img[:, 1], c='blue', s=1, label='Virtual Points')
    
    plt.title('Point Cloud Projection on Image')
    plt.legend()
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
def visualize_depth_map(points_lidar, calib_obj, image_shape, filename='depth_map_viz.png'):
    img_h, img_w = image_shape
    pts_rect = calib_obj.lidar_to_rect(points_lidar)
    pts_img, pts_depth = calib_obj.rect_to_img(pts_rect)
    # 1) 좌표 반올림 (+ 정수형) -> 경계 clip
    xs = np.rint(pts_img[:, 0]).astype(np.int64)
    ys = np.rint(pts_img[:, 1]).astype(np.int64)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    # 2) 양수 깊이만 사용
    valid = (pts_depth > 0)

    # 3) 최근접 깊이 유지: inf로 초기화 후 minimum
    canvas = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(canvas, (ys[valid], xs[valid]), pts_depth[valid])
    canvas[~np.isfinite(canvas)] = 0.0  # 배경 0

    # 4) 시각화용 정규화 (원하는 최대 거리로 스케일)
    max_depth = 80.0  # 필요시 조정
    norm = np.clip(canvas / max_depth, 0.0, 1.0)
    vis_u8 = (norm * 255).astype(np.uint8)

    # 5) 컬러맵
    color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)

    # 6) 배경(깊이 0)은 완전 검정으로 강제
    bg = (canvas == 0)
    color[bg] = (0, 0, 0)

    cv2.imwrite(filename, color)

def project_lidar_to_img_batch(calib_obj, pts_xyz, image_shape):
    H, W = image_shape
    pts_rect = calib_obj.lidar_to_rect(pts_xyz)
    pts_img, pts_depth = calib_obj.rect_to_img(pts_rect)
    valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H) & \
            (pts_depth > 0)
    xs = pts_img[valid, 0].round().astype(np.int32)
    ys = pts_img[valid, 1].round().astype(np.int32)
    xs = np.round(pts_img[:, 0]).astype(int)
    ys = np.round(pts_img[:, 1]).astype(int)
    
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return valid, ys, xs

def flood_on_roi(mono_gray, y, x, patch_size, tol):
    H, W = mono_gray.shape
    ph, pw = patch_size
    sy, ey = max(0, y - ph//2), min(H, y + ph//2)
    sx, ex = max(0, x - pw//2), min(W, x + pw//2)
    roi = mono_gray[sy:ey, sx:ex]
    ry, rx = y - sy, x - sx
    # ROI에서만 flood
    roi_mask = flood(roi, (ry, rx), connectivity=2, tolerance=tol)
    # 원본 크기로 복원
    mask = np.zeros_like(mono_gray, dtype=bool)
    mask[sy:ey, sx:ex] = roi_mask
    return mask

def generate_virtual_points_single_pass(
    mono_depth_rgb, radar_depth_img, seeds_yx, seeds_attrs, 
    calib_info, image_shape,
    points_per_seed=100, patch_size=(150, 200), tol=0.5
):
    H, W = image_shape
    mono_gray = mono_depth_rgb.mean(axis=2)  # (H, W)
    all_virtual = []
    visited = np.zeros((H, W), dtype=bool)
    for (y, x), attrs in zip(seeds_yx, seeds_attrs):
        # flood mask
        if visited[y, x]:
            continue
        region_mask = flood_on_roi(mono_gray, y, x, patch_size, tol)
        
        if not region_mask.any():
            continue
        visited |= region_mask

        # seed의 깊이를 영역 전체에 부여한 지역 깊이맵 생성
        region_depth_map = np.zeros_like(mono_gray, dtype=np.float32)
        region_depth_map[region_mask] = radar_depth_img[y, x]  # seed에서 얻은 깊이

        # 이 seed에 대해 바로 가상 포인트 샘플링
        virt = generate_virtual_points_from_depth_map(
            region_depth_map, calib_info, attrs, num_points=points_per_seed
        )
        if virt.size > 0:
            all_virtual.append(virt)

    if all_virtual:
        return np.vstack(all_virtual)
    return np.empty((0, 8), dtype=np.float32)


def process_single_frame(info, data, image_path, mono_path, calib_path, save_path,
                         points_per_seed = 100, patch_size=(150, 200), tol=0.5, do_visualize=False):
    
    # 데이터 로드: [x, y, z, rcs, v, v', t]
    image = load_image(image_path)
    H, W = image.shape[:2]
    radar_points = np.fromfile(
        os.path.join('./data/vod/radar_5frames/training/velodyne/', info['point_cloud']['lidar_idx'] + '.bin'), 
        dtype=np.float32
    ).reshape(-1, 7)
    mono_depth = load_depth(mono_path)
    calib_obj = Calibration(calib_path)
    calib_info = {
        'calib_obj': calib_obj,
        'calib': {
            'Tr_velo_to_cam': calib_obj.V2C,
            'P2': np.vstack((calib_obj.P2, np.array([[0., 0., 0., 1.]]))),
        }
    }
    calib_obj = calib_info['calib_obj']
    
    # print(radar_points) 여기까지는 제대로 나옴
    # print(calib_info)
    # print(radar_points[:, :3]) 
    radar_img_depth = radar_to_image_depth(calib_info, radar_points, (H,W))
    
    valid, ys, xs = project_lidar_to_img_batch(calib_obj, radar_points[:, :3], (H, W))
    xs, ys = xs[valid], ys[valid]
    
    has_depth = radar_img_depth[ys, xs] > 0
    seed_indices = np.where(valid)[0][has_depth]
    seeds_yx = list(zip(ys[has_depth], xs[has_depth]))
    seeds_attrs = radar_points[seed_indices, 3:] 
     # [rcs, v, v', t]
    virtual_points = generate_virtual_points_single_pass(
        mono_depth, radar_img_depth, seeds_yx, seeds_attrs, calib_info, (H, W),
        points_per_seed=points_per_seed, patch_size=patch_size, tol=tol
    )
    


    # original_with_flag = np.hstack((radar_points, np.zeros((radar_points.shape[0], 1)))) # flag 0: real
    # virtual_with_flag = np.hstack((virtual_points, np.ones((virtual_points.shape[0], 1)))) # flag 1: virtual
    # print(virtual_with_flag.shape)
    # final_point_cloud = np.vstack((radar_points, virtual_points)) # (N, 8)
    if do_visualize:
        os.makedirs(save_path, exist_ok=True)
        idx = info['image']['image_idx']
        visualize_depth_map(radar_points[:, :3], calib_obj, (H, W), filename=os.path.join(save_path, f'original_radar_depth_map_{idx}.png'))
        visualize_depth_map(final_point_cloud[:, :3], calib_obj, (H, W), filename=os.path.join(save_path, f'augmented_point_depth_map_{idx}.png'))

    return radar_points, virtual_points
    
    

def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list

if __name__ == '__main__':
    # 예시 데이터 경로 설정
    # ROOT_DIR = './data/vod/radar_5frames/training'
    # IDX = '08262'
    
    # image_path = os.path.join(ROOT_DIR, 'image_2', f'{IDX}.jpg')
    # radar_path = os.path.join(ROOT_DIR, 'velodyne', f'{IDX}.bin')
    # mono_path = os.path.join(ROOT_DIR, 'depth_viz', f'{IDX}.jpg')
    # calib_path = os.path.join(ROOT_DIR, 'calib', f'{IDX}.txt')
    # save_dir = './visualization_results'
    # os.makedirs(save_dir, exist_ok=True)
    
    
    # calib_obj = Calibration(calib_path)
    # calib_info = {
    #     'calib_obj': calib_obj,
    #     'calib': {
    #         'Tr_velo_to_cam': calib_obj.V2C,
    #         'P2': np.vstack((calib_obj.P2, np.array([[0., 0., 0., 1.]]))),
    #     }
    # }
    
    # # 단일 프레임 처리 및 시각화
    # print(f'Processing frame: {IDX}')
    # args = (IDX, image_path, radar_path, mono_path, save_dir, calib_info)
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    ROOT_PATH = './data/vod/radar_5frames/training'
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor
    dataset = DatasetVoD('data/vod/radar_5frames/kitti_infos_trainval.pkl', predictor)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn = simple_collate,
        pin_memory = False, 
        shuffle = False
    )
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        output_name = info['image']['image_idx'] + '.pkl.npy'
        save_path = './data/vod/radar_5frames/region_growing_points/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_path = os.path.join(save_path, output_name)
        if os.path.exists(output_path):
            LOGGER.info(f"Skipping inference for {output_path} already exists.")
            continue
        index = info['image']['image_idx']
        image_path = os.path.join(ROOT_PATH, 'image_2', f'{index}.jpg')
        mono_path = os.path.join(ROOT_PATH, 'depth_viz', f'{index}.jpg')
        calib_path = os.path.join(ROOT_PATH, 'calib', f'{index}.txt')
        save_path = os.path.join(ROOT_PATH, 'depth_growing')
        real_pts, virtual_pts = process_single_frame(info, data, image_path, mono_path, calib_path, save_path)
        data_dict_new = {
            'virtual_points': virtual_pts,
            'real_points': real_pts
        }
        np.save(output_path, data_dict_new)
    # print(f'Total points in augmented cloud: {total_points}, Virtual points added: {virtual_points_count}')
    # print(f'Visualization saved to: {os.path.join(save_dir, f"{IDX}_augmented_points.png")}')

    
    # LIDAR_DIR = './data/vod/lidar/training'
    # lidar_path = os.path.join(LIDAR_DIR, 'velodyne', f'{IDX}.bin')  
    # print(np.fromfile(lidar_path, dtype=np.float32).shape)  
    # lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    # calib_obj = calib_info['calib_obj']
    # visualize_depth_map(lidar_points[:, :3], calib_obj, (H, W), filename=os.path.join(save_dir, f'lidar_depth_{IDX}.png'))