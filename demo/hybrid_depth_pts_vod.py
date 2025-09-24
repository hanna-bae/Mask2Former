# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from nusc_image_projection import to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torch
import os
from matplotlib import pyplot as plt
import numpy as np
from ml_depth_pro.src.depth_pro import create_model_and_transforms, load_rgb

# Size of the dataset image
H=1216
W=1936
gauss_uniform_ratio = [1, 4]

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
        required=True,
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


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
        
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




def is_within_mask(points_xyc, masks, H=H, W=W):
    seg_mask = masks[:, :-1].reshape(-1, W, H)
    camera_id = masks[:, -1]
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape] # radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

@torch.no_grad()
def add_virtual_mask(masks, labels, points, raw_points, depth_map, num_virtual=100, depth_thresh=10, num_camera=1, intrinsics=None, transforms=None):
    """
    Args:
        masks: instance masks from segmentation network
        labels: labels of each instance mask
        points: lidar points projected to image plane
        raw_points: original lidar points in 3D
        depth_map: monocular depth map for point augmentation
        num_virtual: number of virtual points to generate per instance
        depth_thresh: depth difference threshold
        num_camera: number of cameras
        intrinsics: camera intrinsic matrix
        transforms: lidar to camera transformation matrix
    """
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]] # x, y, z, valid_indicator, camera id

    valid = is_within_mask(points_xyc, masks)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    # remove camera id from masks 
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )
    point_labels = torch.cat([point_labels, raw_points[:, 3:]], dim=1)

    # foreground에 속한 실제 레이더 포인트 마스크
    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

    virtual_points_list = []
    
    # Iterate through each detected instance (mask)
    for idx, mask in enumerate(masks):
        # 1. Get real radar points within this mask
        real_points_in_mask = points[0, torch.where(valid[:, idx])[0], :]
        
        # Check if any real radar points exist in this mask
        if len(real_points_in_mask) > 0:
            # 2. Get the average depth of real radar points in this mask
            # This serves as the reference depth for the object
            avg_depth_from_radar = real_points_in_mask[:, 2].mean()
            #print(mask.shape)
            

            # 3. Find all pixels within the current mask
            mask_indices = mask.reshape(H, W).nonzero()
            #print(mask_indices.shape)
            # 4. Get the corresponding depth values from the depth map
            depth_map = torch.as_tensor(depth_map, device=masks.device, dtype=torch.float32)

            depths_in_mask = depth_map[mask_indices[:, 0], mask_indices[:, 1]]
            
            # 5. Filter pixels based on depth threshold
            # The depth map value should be close to the average depth of the real radar points
            depth_diff = torch.abs(depths_in_mask - avg_depth_from_radar)
            #print(torch.min(depth_diff), torch.max(depth_diff))

            valid_pixels_mask = depth_diff < depth_thresh
            #print("mask_indices shape:", mask_indices.shape)
            #print("valid_pixels_mask shape:", valid_pixels_mask.shape)
            #print("valid_pixels_mask dtype:", valid_pixels_mask.dtype)
            #print("Sum of True:", valid_pixels_mask.sum())

            # Get the indices of the valid pixels
            if valid_pixels_mask.sum() == 0:
                continue
            valid_pixels_indices = mask_indices[valid_pixels_mask]
            
            # 6. Sample virtual points from the valid pixels
            num_valid_pixels = len(valid_pixels_indices)
            if num_valid_pixels > 0:
                num_to_sample = min(num_virtual, num_valid_pixels)
                sampled_indices = torch.randperm(num_valid_pixels, device=masks.device)[:num_to_sample]
                sampled_pixels = valid_pixels_indices[sampled_indices]
                
                # 7. Get the depths for the sampled pixels
                sampled_depths = depth_map[sampled_pixels[:, 0], sampled_pixels[:, 1]].reshape(1, -1)
                
                # 8. Transform sampled 2D pixels to 3D radar coordinates
                # This uses the actual depths from the depth map!
                sampled_pixels_padded = torch.cat(
                    [sampled_pixels.transpose(1, 0).float(),
                     torch.ones((1, sampled_pixels.shape[0]), device=masks.device, dtype=torch.float32)],
                    dim=0
                )

                # Reverse view projection to get 3D camera coordinates
                virtual_points_3d_cam = reverse_view_points(sampled_pixels_padded, sampled_depths, intrinsics[0])
                
                # Transform to 3D Lidar coordinates
                virtual_points_3d_lidar = torch.matmul(torch.inverse(transforms[0]),
                                                torch.cat([
                                                    virtual_points_3d_cam[:3, :],
                                                    torch.ones(1, virtual_points_3d_cam.shape[1], dtype=torch.float32, device=masks.device)
                                                ], dim=0))[:3]
                
                # Add instance ID and labels
                instance_id = torch.tensor(idx, dtype=torch.float32, device=masks.device).reshape(1, 1).repeat(sampled_pixels.shape[0], 1)
                
                # Take label from real points within the mask
                sampled_labels = point_labels[torch.where(valid[:, idx])[0][0], :].unsqueeze(0).repeat(sampled_pixels.shape[0], 1)
                
                virtual_points_instance = torch.cat([virtual_points_3d_lidar.transpose(1,0), sampled_labels], dim=-1)
                
                virtual_points_list.append(virtual_points_instance)

    if len(virtual_points_list) == 0:
        return None, None, None, None

    all_virtual_points = torch.cat(virtual_points_list, dim=0)
    all_real_points_in_foreground = raw_points[foreground_real_point_mask.bool(), :]
    
    # Create final output tensor for real points
    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)[..., :11]
    real_point_labels = torch.max(real_point_labels, dim=0)[0]
    all_real_points = torch.cat([all_real_points_in_foreground, real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1)


def postprocess(res):
    result = res['instances']
    #print(result.pred_classes.shape) # 100
    #print(result.pred_masks.shape) # (100, 1216, 1936)
    #print(result.scores.shape) # 100
    labels = result.pred_classes
    scores = result.scores 
    masks = result.pred_masks.reshape(scores.shape[0], W*H)  # 类别数量 * pixel数量
    boxes = result.pred_boxes.tensor.to(masks.device)

    # remove empty mask and their scores / labels 
    empty_mask = masks.sum(dim=1) == 0

    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]
    boxes = boxes[~empty_mask]
    masks = masks.reshape(-1, H, W).permute(0, 2, 1).reshape(-1, W*H)
    return labels, scores, masks


def read_file(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    return points

def test_valid(labels):
    class2index = {
        'person': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motorcycle': 6,
        'bicycle': 7
    }
    selected_class = ['car', 'person', 'rider', 'bicycle', 'motorcycle']
    index = [class2index[x] for x in selected_class]
    output = torch.zeros_like(labels)
    for idx, label in enumerate(labels):
        if label in index:
            output[idx] = 1
    return output > 0.5

# add ml-depth-pro
def get_depth_map(image_path, device):

    img, _, f_px = load_rgb(image_path)
    # Load model
    depth_model, transform= create_model_and_transforms(
        device=device,
        precision=torch.half,
    ) 
    depth_model.eval()
    #print(image.shape)
    prediction = depth_model.infer(transform(img), f_px=f_px)

    # Extrat the depth
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    return depth 



@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=1):
    #print(data[-1])
    all_cams_from_lidar = [info['calib']['Tr_velo_to_cam']] # 4*4
    all_cams_intrinsic = [info['calib']['P2'][:3, :3]] # 3*3

    pts_name = info['point_cloud']['lidar_idx'] + '.bin'
    pts_path = os.path.join('./data/vod/training/velodyne/', pts_name)
    img_path = os.path.join('./data/vod/training/image_2/', info['image']['image_idx'] + '.jpg')
    lidar_points = read_file(pts_path)

    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0)

    masks = [] 
    labels = [] 
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1)

    #print(data[-1]["image"].shape)
    depth_map = get_depth_map(img_path, device='cuda:0')
    depth_map = depth_map[:H]
    # ==================================

    # semantic mask
    #print(type(data), data)
    result = predictor.model(data[1:])

    for camera_id in range(num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])

        camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, camera_id], dim=1)
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1) # 转成one hot形式

        masks.append(pred_mask)
        labels.append(transformed_labels)
    
    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)
    # lidar投影到图像上
    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic),
                     H=H, W=W)
    camera_ids = torch.arange(1, dtype=torch.float32, device='cuda:0').reshape(1, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1)
    k = 2
    if len(masks) == 0:
        res = None
    else:
        # === [변경: depth_map 인자 추가] ===
        res = add_virtual_mask(masks, labels, P, to_tensor(lidar_points), depth_map,
             intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar))
    # ==================================

    if res is not None and res[0] is not None:
        virtual_points, foreground_real_points, foreground_indices = res 
        return virtual_points.cpu().numpy(), foreground_real_points.cpu().numpy(), foreground_indices.cpu().numpy()
    else:
        return np.zeros([0, 14]), np.zeros([0, 19]), np.zeros(0)

def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    predictor = demo.predictor
    data_loader = DataLoader(
        Datasetvod('data/vod/kitti_infos_trainval.pkl', predictor),
        batch_size=1,
        num_workers=0,
        collate_fn=simple_collate,
        pin_memory=False,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        output_name = info['image']['image_idx'] + '.pkl.npy'
        save_path = './data/vod/training/depth_mask/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        output_path = os.path.join(save_path, output_name)
        if os.path.exists(output_path):
            print(f'[INFO] {output_path} already exists. Skipping...')
            continue
        
        res = process_one_frame(info, predictor, data)

        if res is not None:
            virtual_points, real_points, indices = res
        else:
            virtual_points = np.zeros([0, 14])
            real_points = np.zeros([0, 19])
            indices = np.zeros(0)

        virtual_points_result = np.concatenate([virtual_points[:, :3], virtual_points[:, -5:], virtual_points[:, 3:11]], axis=1)
        real_points_result = np.concatenate([real_points[:, :8], real_points[:, 8:]], axis=1)
        data_dict_new = {
            'virtual_points': virtual_points_result,
            'real_points': real_points_result
        }

        np.save(output_path, data_dict_new)