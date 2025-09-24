import argparse
import glob
import multiprocessing as mp
import os
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from matplotlib import pyplot as plt
import numpy as np
import torch
import matplotlib.patheffects as pe
import cv2 

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

# Size of the dataset image
H=810
W=1280
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

class DatasetTJ4D(Dataset):
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

        img_name = info['image']['image_idx'] + '.png'
        img_path = os.path.join('./data/tj4d/training/image_2/', img_name)

        original_image = cv2.imread(img_path)
        if original_image is None:
            raise FileNotFoundError(f"[ERROR] image not found: {img_path}")
        original_image = original_image[:810]

        if self.predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.predictor.aug.get_transform(original_image).apply_image(original_image) # 900 * 1600 -> 506 * 900 for nuScenes
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
def add_virtual_mask(masks, labels, points, raw_points, num_virtual=100, dist_thresh=3000, num_camera=1, intrinsics=None, transforms=None, k=5):
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]] # x, y, z, valid_indicator, camera id

    valid = is_within_mask(points_xyc, masks)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    # remove camera id from masks 
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )  
    point_labels = torch.concat([point_labels, raw_points[:, 3:]], dim=1)

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

    gauss_shape = 51
    gauss = gaussian_2d([gauss_shape, gauss_shape], sigma=7)
    gauss = torch.from_numpy(gauss).to(points)

    offsets = [] 
    prob_map_all = torch.zeros_like(masks[0].reshape(W, H))
    for idx, mask in enumerate(masks):
        if len(torch.where(valid[:, idx])[0]) != 0:
            cur_points = points[0, torch.where(valid[:, idx])[0], :]
            mask = mask.reshape(W, H)
            prob_map = torch.zeros_like(mask)
            xr, yr = gauss_shape, gauss_shape
            xr, yr = round(xr / 2), round(yr / 2)
            for point in cur_points:
                x, y = int(point[0]), int(point[1])

                x1, x2 = x - xr, x + xr - 1
                y1, y2 = y - yr, y + yr - 1

                # filter out of range
                x1 = max(x1, 0)
                y1 = max(y1, 0)

                x2 = min(W, x2)
                y2 = min(H, y2)

                prob_map[x1:x2, y1:y2] = prob_map[x1:x2, y1:y2] + gauss[x1 - x + xr:x2 - x + xr, y1 - y + yr:y2 - y + yr]
            # normalize and filter point outside mask
            indices = mask.nonzero()

            prob_map = prob_map * mask 
            prob_map_all = prob_map_all + prob_map
            base = prob_map[indices[:, 0], indices[:, 1]] + 1e-6
            selected_indices_gauss = torch.multinomial(base, num_virtual // 2, replacement=True)

            uni_prob = torch.ones_like(base)
            uni_prob[selected_indices_gauss] = 0
            if int(torch.sum(uni_prob)) == 0:
                selected_indices = selected_indices_gauss
            else:
                selected_indices_uni = torch.multinomial(uni_prob, min(int(torch.sum(uni_prob)), num_virtual // 2))
                selected_indices = torch.concat([selected_indices_gauss, selected_indices_uni])
        else:
            indices = mask.reshape(W, H).nonzero()
            selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual] # 从mask中挑取采样点
        if len(selected_indices) < num_virtual:
                selected_indices = torch.cat([selected_indices, selected_indices[
                    selected_indices.new_zeros(num_virtual-len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)

    offsets = torch.stack(offsets, dim=0)
    virtual_point_instance_ids = torch.arange(1, 1+masks.shape[0], 
        dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)

    virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3) # 每个采样点在哪个mask上
    virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)

    valid_mask = valid.sum(dim=1)>0
    real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1)

    # avoid matching across instances 在计算距离时，不同instance（mask）的距离会显著增大
    real_points[:, -1] *= 1e4 
    virtual_points[:, -1] *= 1e4 

    if len(real_points) == 0:
        return None 
    
    k_all_virtual_points = []
    k_all_real_points = []
    k_all_point_labels = []
    for idx, k_phase in enumerate(gauss_uniform_ratio):
        if idx == 0: # gauss
            cur_mask = (torch.arange(0, len(masks))*num_virtual).unsqueeze(0) + torch.arange(0, num_virtual // 2).unsqueeze(1)
            cur_mask = cur_mask.reshape([-1])
        elif idx == 1: # uniform
            cur_mask = (torch.arange(0, len(masks))*num_virtual).unsqueeze(0) + torch.arange(num_virtual // 2, num_virtual).unsqueeze(1)
            cur_mask = cur_mask.reshape([-1])
        else:
            raise NotImplementedError

        cur_virtual_points = virtual_points[cur_mask]
        cur_virtual_point_camera_ids = virtual_point_camera_ids[cur_mask]

        dist = torch.norm(cur_virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1) # N_virtual * N_real
        k_min = min(k_phase, real_points.shape[0])
        k_nearest_dist, k_nearest_indices = torch.topk(dist, k=k_min, dim=1, largest=False) # N_virtual
        for j in range(k_min):
            nearest_dist = k_nearest_dist[:, j]
            nearest_indices = k_nearest_indices[:, j]
            mask = nearest_dist < dist_thresh 

            indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

            nearest_indices = indices[nearest_indices[mask]]
            virtual_points_ = cur_virtual_points[mask]
            virtual_point_camera_ids_ = cur_virtual_point_camera_ids[mask]
            all_virtual_points = [] 
            all_real_points = [] 
            all_point_labels = []

            for i in range(num_camera):
                camera_mask = (virtual_point_camera_ids_ == i).squeeze()
                per_camera_virtual_points = virtual_points_[camera_mask]
                per_camera_indices = nearest_indices[camera_mask]
                per_camera_virtual_points_depth = points.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1)

                per_camera_virtual_points = per_camera_virtual_points[:, :2] # remove instance id 
                per_camera_virtual_points_padded = torch.cat(
                        [per_camera_virtual_points.transpose(1, 0).float(), 
                        torch.ones((1, len(per_camera_virtual_points)), device=per_camera_indices.device, dtype=torch.float32)],
                        dim=0
                    )
                # 将real point的depth分布到virtual point上，并将其转换回lidar 坐标系（3d空间）
                per_camera_virtual_points_3d = reverse_view_points(per_camera_virtual_points_padded, per_camera_virtual_points_depth, intrinsics[i])

                per_camera_virtual_points_3d[:3] = torch.matmul(torch.inverse(transforms[i]),
                        torch.cat([
                                per_camera_virtual_points_3d[:3, :], 
                                torch.ones(1, per_camera_virtual_points_3d.shape[1], dtype=torch.float32, device=per_camera_indices.device)
                            ], dim=0)
                    )[:3]

                all_virtual_points.append(per_camera_virtual_points_3d.transpose(1, 0))
                all_real_points.append(raw_points.reshape(1, -1, 8).repeat(num_camera, 1, 1).reshape(-1, 8)[per_camera_indices])
                all_point_labels.append(point_labels[per_camera_indices])
            k_all_virtual_points.append(all_virtual_points)
            k_all_real_points.append(all_real_points)
            k_all_point_labels.append(all_point_labels)

    k_all_virtual_points = [x[0] for x in k_all_virtual_points]
    k_all_real_points = [x[0] for x in k_all_real_points]
    k_all_point_labels = [x[0] for x in k_all_point_labels]

    all_virtual_points = torch.cat(k_all_virtual_points, dim=0)
    all_real_points = torch.cat(k_all_real_points, dim=0)
    all_point_labels = torch.cat(k_all_point_labels, dim=0)

    all_virtual_points = torch.cat([all_virtual_points, all_point_labels], dim=1) # 位置_3 + 类别_10 + 置信度_1 + 雷达特征_4 = 18

    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)[..., :11] # 类别_10 + 置信度_1 = 11
    real_point_labels  = torch.max(real_point_labels, dim=0)[0]
    # 位置_3 + 雷达特征_4 + 类别_10 + 置信度_1 = 18
    all_real_points = torch.cat([raw_points[foreground_real_point_mask.bool(), :], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1), prob_map_all

def postprocess(res):
    result = res['instances']
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
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 8)
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

# add 
def augment_with_3d_gaussan(points, num_new_points=10, rcs_scale=1, vel_scale=0.1):
    '''
    Args: 
        points (np.ndarray): (N, 19) position 3 + class 8 + zero padding 2 + confidence 1 + radar 5
        num_new_points (int): number of new points to generate
        rcs_scale (float): scale for RCS
        vel_scale (float): scale for velocity
    returns:
        position 3 + class 8 + zero padding 2 + confidence 1 + radar 5
    '''
    augmented_points = []
    for point in points:
        pos = point[:3]  # position (x, y, z)
        rcs = point[15]
        vx_comp = point[16]  # velocity x component
        vy_comp = point[17]  # velocity y component

        sigma_rcs = rcs_scale * np.log(1+ np.abs(rcs))
        if sigma_rcs < 1e-6:
            sigma_rcs = 0.5

        cov_matrix = np.eye(3) * sigma_rcs 

        velocity_shift = np.array([vx_comp, vy_comp, 0]) * vel_scale
        mean_vector = pos + velocity_shift

        new_points = np.random.multivariate_normal(
            mean=mean_vector, 
            cov=cov_matrix, 
            size=num_new_points
        )

        attributes = np.tile(point[3:], (num_new_points, 1))  # copy attributes (class, zero padding, confidence)
        new_points = np.hstack((new_points, attributes))  # combine position and attributes
        augmented_points.append(new_points)
    return np.vstack(augmented_points) if augmented_points else np.array([])

@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=1):
    all_cams_from_lidar = [info['calib']['Tr_velo_to_cam']] # 4*4
    all_cams_intrinsic = [info['calib']['P2'][:3, :3]] # 3*3

    pts_name = info['point_cloud']['lidar_idx'] + '.bin'
    pts_path = os.path.join('./data/tj4d/training/velodyne/', pts_name)
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
        res  = add_virtual_mask(masks, labels, P, to_tensor(lidar_points), 
            intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar), k=k)
    
    if res is not None:
        virtual_points, foreground_real_points, foreground_indices, prob_map_all = res 
        # augment with 3D gaussian
        augmented_points = augment_with_3d_gaussan(foreground_real_points.cpu().numpy(), num_new_points=5, rcs_scale=1, vel_scale=0.1)
        virtual_points = np.concatenate([virtual_points.cpu().numpy(), augmented_points], axis=0)
        return augmented_points, foreground_real_points.cpu().numpy(), foreground_indices.cpu().numpy(), prob_map_all.T.cpu().numpy()
    else:
        return None 

def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    predictor = demo.predictor
    dataset = DatasetTJ4D('data/tj4d/kitti_infos_trainval.pkl', predictor)
    print("[DEBUG] dataset length: ", len(dataset)) # 제대로 뜸. 
    # 오류나서 pin memory False -> True로 변경
    data_loader = DataLoader(
        DatasetTJ4D('data/tj4d/kitti_infos_trainval.pkl', predictor),
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
        #save_path = './data/tj4d/training/mask_maskformer_with_label_k_1_gauss_k_4_uniform/'
        save_path = './data/3d_viz_new/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        
        res = process_one_frame(info, predictor, data)

        if res is not None:
            virtual_points, real_points, indices, prob_map_all = res # virtual_points: 位置_3 + 类别_8 + 置信度_1 + 雷达特征_5
            img_path = os.path.join('./data/tj4d/training/image_2/', info['image']['image_idx'] + '.png')
            img = cv2.imread(img_path)
        else:                                          # real_points:    位置_3 + 雷达特征_5 + 类别_8 + 置信度_1
            virtual_points = np.zeros([0, 14])
            real_points = np.zeros([0, 15])
            indices = np.zeros(0)

        
        # visualize bev 
        plt.clf()
        fig, ax = plt.subplots(dpi=500)
        gca = plt.gca()
        gca.set_xlim(0, 69.12)
        gca.set_ylim(-39.68, 39.68)

        x, y = real_points[:, 0], real_points[:, 1]
        for i in range(len(x)):
            circle = plt.Circle([x[i], y[i]], .2, facecolor='blue') # blue (real) 
            ax.add_artist(circle)
            stroke_effect = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
            circle.set_path_effects(stroke_effect)
        
        x_, y_ = virtual_points[:, 0], virtual_points[:, 1]
        for i in range(len(x_)):
            circle = plt.Circle([x_[i], y_[i]], .2, facecolor='red') # read (virtual) 
            ax.add_artist(circle)
            stroke_effect = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
            circle.set_path_effects(stroke_effect)

        plt.axis('off')
        plt.savefig(save_path+(f"/{info['image']['image_idx']}_bev.png"))

        pass