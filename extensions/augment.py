import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import open3d as o3d

from emd_ import emd_module
EMD = emd_module.emdModule()

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds within [-15, 15] degrees around z axis
        Input:
          B x N x 3 array
        Return:
          B x N x 3 array
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    # degree to radian conversion
    min_deg = -15.0
    max_deg = 15.0

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform(min_deg, max_deg) * np.pi / 180.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = np.array([
            [cosval,  sinval, 0],
            [-sinval, cosval, 0],
            [0,       0,      1]
        ], dtype=np.float32)

        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def jitter_point_cloud_torch(batch_data, sigma=0.005, clip=0.005):
    # batch_data  B N 3  (torch tensor)
    B, N, C = batch_data.shape
    device = batch_data.device

    assert clip > 0

    # gaussian noise 생성 (N(0, sigma^2))
    jitter = torch.randn(B, N, C, device=device) * sigma

    # clip [-clip, clip]
    jitter = torch.clamp(jitter, -clip, clip)

    # 원본 포인트에 더하기
    return batch_data + jitter

def jitter_point_cloud_torch_partial(batch_data, sigma=0.005, clip=0.03, ratio=0.7):
    # batch_data  B N 3
    B, N, C = batch_data.shape
    device = batch_data.device

    assert clip > 0
    assert 0 <= ratio <= 1

    # clone so original isn't modified
    data = batch_data.clone()

    # jitter count
    n_jitter = int(N * ratio)

    for b in range(B):
        # 선택된 포인트 index
        idx = torch.randperm(N, device=device)[:n_jitter]

        # gaussian noise N(0, sigma^2)
        jitter = torch.randn(n_jitter, C, device=device) * sigma
        jitter = torch.clamp(jitter, -clip, clip)

        data[b, idx, :] += jitter

    return data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_scale_point_cloud_torch(batch_data, scale_low=0.8, scale_high=1.2):
    # batch_data  B N 3  (torch tensor)
    B, N, C = batch_data.shape
    device = batch_data.device

    # B개의 scale을 uniform 분포에서 샘플링
    scales = (torch.rand(B, device=device) * (scale_high - scale_low)) + scale_low

    # 각 배치별로 스케일 적용
    batch_data = batch_data * scales.view(B, 1, 1)

    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_point_dropout_torch(batch_pc, max_dropout_ratio=0.875):
    # batch_pc  B N 3  (torch tensor)
    B, N, C = batch_pc.shape
    device = batch_pc.device

    for b in range(B):
        dropout_ratio = torch.rand(1, device=device).item() * max_dropout_ratio
        mask = torch.rand(N, device=device) <= dropout_ratio  # True means drop
        
        if mask.any():
            batch_pc[b, mask] = batch_pc[b, 0].clone()

    return batch_pc

# -----------------------------
# PointMixup with EMD matching
# -----------------------------
def pointmixup_emd(pc, lab, alpha=0.2):
    # pc: B x 3 x N
    # lab: B 또는 B x C
    B, C, N = pc.shape

    # 1) permutation
    perm = torch.randperm(B)
    pc2 = pc[perm]            # B x 3 x N
    lab2 = lab[perm]

    # 2) transpose for OT: B x N x 3
    x = pc.transpose(1, 2)
    y = pc2.transpose(1, 2)

    _, ass = EMD(x, y, 0.005, 300)
    ass = ass.long()
    for i in range(B):
        y[i] = y[i][ass[i]]

    # 3) EMD-based matching → transport plan T (B x N x N)
    matched_y = y.transpose(1, 2)

    # 5) sample lambda
    lam_np = 0.5 - np.abs(np.random.beta(alpha, alpha, B) - 0.5)
    lam = torch.from_numpy(lam_np).float().to(pc.device)  # B
    lam_pc = lam.reshape(B, 1, 1)

    # 6) mix
    mixed_pc = lam_pc * pc + (1 - lam_pc) * matched_y

    # 7) mix labels
    if lab.dim() == 1:
        num_classes = 15
        lab_oh = F.one_hot(lab, num_classes=num_classes).float()
        lab2_oh = F.one_hot(lab2, num_classes=num_classes).float()
        lam_lab = lam.view(B, 1)
        mixed_lab = lam_lab * lab_oh + (1 - lam_lab) * lab2_oh
    else:
        lam_lab = lam.view(B, 1)
        mixed_lab = lam_lab * lab + (1 - lam_lab) * lab2

    return mixed_pc, mixed_lab