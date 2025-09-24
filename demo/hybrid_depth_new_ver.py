# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import argparse 
import torch 
import cv2 

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
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
from ml_depth_pro.src.depth_pro import create_model_and_transforms, load_rgb

# Size of the dataset image
H=1216
W=1936

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

_DEPTH_CACHE = {"model": None, "transform": None, "device": None}

def get_depth_model(device="cuda:0", precision=torch.half):
    if _DEPTH_CACHE["model"] is None:
        model, transform = create_model_and_transforms(device=device, precision=precision)
        model.eval()
        _DEPTH_CACHE.update({"model": model, "transform": transform, "device": device})
    return _DEPTH_CACHE["model"], _DEPTH_CACHE["transform"], _DEPTH_CACHE["device"]

def is_within_mask(points_xyc, masks, H=H, W=W):
    seg_mask = masks[:, :-1].reshape(-1, W, H)
    camera_id = masks[:, -1]
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

def get_depth_map(image_path, device):
    # 기존: 매 호출마다 create_model_and_transforms() => 제거
    img, _, f_px = load_rgb(image_path)
    depth_model, transform, dev = get_depth_model(device=device, precision=torch.half)
    with torch.no_grad():
        pred = depth_model.infer(transform(img), f_px=f_px)
    depth = pred["depth"].detach().cpu().numpy().squeeze()
    return depth
def sample_pixels_from_mask(mask: torch.Tensor,
                            max_pixels_per_mask: int = 2000,
                            stride: int = 4,
                            edge_ratio: float = 0.3):
    """
    mask: (H, W) bool/0-1
    리턴: (M, 2) [y, x]
    """
    H, W = mask.shape
    # 1) stride 그리드 샘플
    if stride > 1:
        grid = torch.zeros_like(mask, dtype=torch.bool)
        grid[::stride, ::stride] = True
        mask_s = mask & grid
    else:
        mask_s = mask

    ys, xs = torch.nonzero(mask_s, as_tuple=True)
    if ys.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=mask.device)

    # 2) 옵션: 엣지 가중 샘플(간단히 마스크 경계 근처를 조금 더 뽑기)
    # 비용을 최소화하려고 4-이웃 차분으로 근사
    if edge_ratio > 0 and ys.numel() > 0:
        # 경계 근사
        pad = torch.nn.functional.pad(mask.float(), (1,1,1,1), value=0)
        # 라플라시안 근사로 경계 추출
        kern = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=mask.device, dtype=torch.float32)
        kern = kern.view(1,1,3,3)
        edge = torch.nn.functional.conv2d(pad[None,None], kern, padding=0).abs()[0,0]
        edge = (edge > 0).to(torch.bool)
        edge_ys, edge_xs = torch.nonzero(edge, as_tuple=True)

        # 교집합
        edge_flags = torch.zeros(ys.shape[0], dtype=torch.bool, device=mask.device)
        if edge_ys.numel() > 0:
            # 빠르게 근사: 해시 인덱스
            key_all = ys * W + xs
            key_edge = edge_ys * W + edge_xs
            sel = torch.isin(key_all, key_edge)
            edge_flags = sel

        # edge/non-edge 분리 샘플링
        n_total = min(max_pixels_per_mask, ys.numel())
        n_edge = int(n_total * edge_ratio)
        n_plain = n_total - n_edge

        idx = torch.arange(ys.shape[0], device=mask.device)
        edge_idx = idx[edge_flags]
        plain_idx = idx[~edge_flags]

        # 랜덤 셔플 후 뽑기
        if edge_idx.numel() > 0:
            edge_sel = edge_idx[torch.randperm(edge_idx.numel(), device=mask.device)[:min(n_edge, edge_idx.numel())]]
        else:
            edge_sel = torch.empty(0, dtype=torch.long, device=mask.device)
        if plain_idx.numel() > 0:
            plain_sel = plain_idx[torch.randperm(plain_idx.numel(), device=mask.device)[:min(n_plain, plain_idx.numel())]]
        else:
            plain_sel = torch.empty(0, dtype=torch.long, device=mask.device)

        sel = torch.cat([edge_sel, plain_sel], dim=0)
        ys, xs = ys[sel], xs[sel]
    else:
        # 단순 랜덤 상한
        if ys.numel() > max_pixels_per_mask:
            idx = torch.randperm(ys.numel(), device=mask.device)[:max_pixels_per_mask]
            ys, xs = ys[idx], xs[idx]

    return torch.stack([ys, xs], dim=1)  # (M, 2)
@torch.no_grad()
def add_pixel_virtual_points(
    masks, origin_mask, labels, points, raw_points,
    num_camera=1, intrinsics=None, transforms=None, depth_map=None,
    max_pixels_per_mask=2000, stride=4, edge_ratio=0.3,
    knn_k=3,  # k를 5 -> 3으로 줄여도 충분한 경우가 많음
    max_virtual_points_per_frame=30000  # 프레임 전체 상한
):
    """
    반환:
      all_virtual_points: (N_v, 18) 또는 None
      all_real_points:    (N_r, 18) 또는 None
    """
    if raw_points.shape[0] == 0 or masks.shape[0] == 0:
        return None, None

    device = points.device
    H, W = depth_map.shape

    # 미리 2D lidar 좌표만 가져오기 (배치 kNN용)
    # points: (N_pts, 5) -> [:, :2] = (x, y)
    pts_xy = points.reshape(-1, 5)[:, :2]  # (N_pts, 2)

    all_pix_xy = []
    all_lbls = []
    # 마스크별 픽셀 샘플
    for mask_idx, mask in enumerate(masks):
        pix_yx = sample_pixels_from_mask(
            mask.bool(),
            max_pixels_per_mask=max_pixels_per_mask,
            stride=stride,
            edge_ratio=edge_ratio
        )
        if pix_yx.numel() == 0:
            continue
        # [y, x] -> [x, y] 로 바꿔둠 (코드 일관성)
        pix_xy = torch.stack([pix_yx[:,1], pix_yx[:,0]], dim=1).to(device).float()  # (M, 2)
        all_pix_xy.append(pix_xy)
        all_lbls.append(labels[mask_idx].unsqueeze(0).repeat(pix_xy.shape[0], 1))  # (M, 11)

    if len(all_pix_xy) == 0:
        return None, None

    all_pix_xy = torch.cat(all_pix_xy, dim=0)  # (Q, 2)
    all_lbls = torch.cat(all_lbls, dim=0)      # (Q, 11)

    # depth 추출 (벡터화)
    xs = all_pix_xy[:,0].long().clamp_(0, W-1)
    ys = all_pix_xy[:,1].long().clamp_(0, H-1)
    z = depth_map[ys, xs]  # (Q,)

    # 너무 작은/이상치 깊이는 버림
    valid_z = torch.isfinite(z) & (z > 1e-3) & (z < 300.0)  # 장면 범위에 맞춰 조정
    if valid_z.sum() == 0:
        return None, None
    all_pix_xy = all_pix_xy[valid_z]
    all_lbls   = all_lbls[valid_z]
    z          = z[valid_z]

    # 배치 kNN: 픽셀 -> LiDAR 포인트(2D) 거리, chunk로 나눠 topk
    Q = all_pix_xy.shape[0]
    K = knn_k
    # 메모리 고려해 chunk 처리
    chunk = 8192
    knn_features = []
    for s in range(0, Q, chunk):
        e = min(s+chunk, Q)
        q = all_pix_xy[s:e]  # (C, 2)
        dists = torch.cdist(q, pts_xy)  # (C, N_pts)
        dvals, didx = torch.topk(dists, k=min(K, pts_xy.shape[0]), largest=False, dim=1)  # (C, K)
        # 레이더 특성 평균(가중치 1/d)
        w = 1.0 / (dvals + 1e-6)                 # (C, K)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        neigh_feat = raw_points[didx, 3:7]       # (C, K, 4)
        avg_feat = (w.unsqueeze(-1) * neigh_feat).sum(dim=1)  # (C, 4)
        knn_features.append(avg_feat)
    knn_features = torch.cat(knn_features, dim=0)  # (Q, 4)

    # 역투영(벡터화) : x = (u - cx) * z / fx, y = (v - cy) * z / fy
    Kcam = intrinsics[0]  # (3,3)
    fx, fy = Kcam[0,0], Kcam[1,1]
    cx, cy = Kcam[0,2], Kcam[1,2]
    u = all_pix_xy[:,0]
    v = all_pix_xy[:,1]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    # cam -> lidar 변환(벡터화)
    T = transforms[0]            # (4,4) : lidar->cam (보통)
    Tinv = torch.inverse(T)      # cam->lidar
    ones = torch.ones_like(z_cam)
    pts_cam_h = torch.stack([x_cam, y_cam, z_cam, ones], dim=0)  # (4, Q)
    pts_lidar = (Tinv @ pts_cam_h)[:3, :].T                      # (Q, 3)

    # virtual point 조립: [lidar_xyz(3), avg_feat(4), labels(11)] = 18
    all_virtual = torch.cat([pts_lidar, knn_features, all_lbls], dim=1)  # (Q, 18)

    # 프레임 상한 적용(라지 씬 대비 안전)
    if all_virtual.shape[0] > max_virtual_points_per_frame:
        idx = torch.randperm(all_virtual.shape[0], device=all_virtual.device)[:max_virtual_points_per_frame]
        all_virtual = all_virtual[idx]

    # ===== 아래 real point 라벨링은 기존 로직 유지 =====
    points_xyc = points.reshape(-1, 5)[:, [0,1,4]]
    valid = is_within_mask(points_xyc, origin_mask)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )
    point_labels = torch.concat([point_labels, raw_points[:, 3:]], dim=1)

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)[..., :11]
    real_point_labels = torch.max(real_point_labels, dim=0)[0]
    all_real_points = torch.cat([raw_points[foreground_real_point_mask.bool(), :], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual, all_real_points

def postprocess(res):
    result = res['instances']
    labels = result.pred_classes # 예상 차원: (N_instances)
    scores = result.scores # 예상 차원: (N_instances)
    masks = result.pred_masks # 예상 차원: (N_instances, H, W)
    masks_2 = result.pred_masks.reshape(scores.shape[0], W*H)
    boxes = result.pred_boxes.tensor.to(masks.device) # 예상 차원: (N_instances, 4)

    # 마스크가 비어있는 경우(픽셀 합이 0) 해당 마스크를 제거합니다.
    empty_mask = masks.sum(dim=[1, 2]) == 0 # 예상 차원: (N_instances)
    empty_mask_2 = masks_2.sum(dim=1) == 0

    labels = labels[~empty_mask] # 예상 차원: (N_non_empty)
    scores = scores[~empty_mask] # 예상 차원: (N_non_empty)
    masks = masks[~empty_mask] # 예상 차원: (N_non_empty, H, W)
    masks_2 = masks_2[~empty_mask_2] # 예상 차원: (N_non_empty, H*W)
    boxes = boxes[~empty_mask] # 예상 차원: (N_non_empty, 4)
    masks_2 = masks_2.reshape(-1, W, H).permute(0, 2, 1).reshape(-1, W*H)
    
    # is_within_mask에서 올바른 좌표를 얻기 위해 (N, H, W) 형태로 반환
    return labels, scores, masks, masks_2

def read_file(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    return points

def test_valid(labels):
    class2index = {
        'person': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7
    }
    selected_class = ['car', 'person', 'rider', 'bicycle', 'motorcycle']
    index = [class2index[x] for x in selected_class]
    output = torch.zeros_like(labels)
    for idx, label in enumerate(labels):
        if label in index:
            output[idx] = 1
    return output > 0.5


@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=1):
    all_cams_from_lidar = [info['calib']['Tr_velo_to_cam']] # 4x4 행렬
    all_cams_intrinsic = [info['calib']['P2'][:3, :3]] # 3x3 행렬
    
    img_name = info['image']['image_idx'] + '.jpg'
    img_path = os.path.join('./data/vod/training/image_2/', img_name)
    
    # 1. 제로샷 깊이 맵 생성
    depth_map = get_depth_map(img_path, device='cuda:0')
    depth_map = torch.from_numpy(depth_map).to('cuda:0') # 예상 차원: (H, W)

    # 2. LiDAR 포인트 로드
    pts_name = info['point_cloud']['lidar_idx'] + '.bin'
    pts_path = os.path.join('./data/vod/training/velodyne/', pts_name)
    lidar_points = to_tensor(read_file(pts_path)) # 예상 차원: (N, 7)

    # 3. Mask2Former를 통해 마스크, 레이블, 스코어 획득
    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)
    one_hot_labels = torch.stack(one_hot_labels, dim=0) # 예상 차원: (10, 10)

    masks = [] 
    labels = [] 
    result = predictor.model(data[1:])
    mask_2 = []

    for camera_id in range(num_camera):
        pred_label, score, pred_mask, pred_mask_2 = postprocess(result[camera_id]) # 예상 차원: (N_masks), (N_masks), (N_masks, H, W)
        
        camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
        pred_mask_2 = torch.cat([pred_mask_2, camera_id], dim=1)
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10)) # 예상 차원: (N_masks, 10)
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1) # 예상 차원: (N_masks, 11)

        masks.append(pred_mask)
        mask_2.append(pred_mask_2)
        labels.append(transformed_labels)
    
    masks = torch.cat(masks, dim=0) # 예상 차원: (N_total_masks, H, W)
    labels = torch.cat(labels, dim=0) # 예상 차원: (N_total_masks, 11)
    mask_2 = torch.cat(mask_2, dim=0) # 예상 차원: (N_total_masks, H*W + 1)
    
    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic), H=H, W=W) # 예상 차원: (N_lidar, 5)
    camera_ids = torch.arange(1, dtype=torch.float32, device='cuda:0').reshape(1, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1)
    # 4. 픽셀 기반 가상 포인트 생성 함수 호출
    if len(masks) == 0:
        res = None
    else:
        
        res = add_pixel_virtual_points(
    masks, mask_2, labels, P, to_tensor(lidar_points),
    intrinsics=to_batch_tensor(all_cams_intrinsic),
    transforms=to_batch_tensor(all_cams_from_lidar),
    depth_map=depth_map,
    # 다운샘플/서브샘플 파라미터(씬에 맞춰 튜닝)
    max_pixels_per_mask=1000,
    stride=6,
    edge_ratio=0.3,
    knn_k=3,
    max_virtual_points_per_frame=20000
)

    if res is not None:
        virtual_points, real_points = res 
        return virtual_points.cpu().numpy(), real_points.cpu().numpy()
    else:
        return np.zeros([0, 18]), np.zeros([0, 18])


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
        save_path = './data/vod/training/vod_hybrid_depth_new_ver/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        output_path = os.path.join(save_path, output_name)
        
        virtual_points, real_points = process_one_frame(info, predictor, data)

        virtual_points_result = np.concatenate([virtual_points[:, :3], virtual_points[:, -4:], virtual_points[:, 3:11]], axis=1)
        real_points_result = real_points[:, :15]
        
        data_dict_new = {
            'virtual_points': virtual_points_result,
            'real_points': real_points_result
        }

        np.save(output_path, data_dict_new)