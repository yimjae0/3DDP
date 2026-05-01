import math
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pointops.functions import pointops
from pointnet.pointnetpp_util import index_points


def preprocess_align(pc_batch):
    """Align a batch of point clouds via centroid + PCA rotation.

    Args:
        pc_batch: (B, 3, N)
    Returns:
        aligned batch of same shape
    """
    B, _, N = pc_batch.shape
    aligned = []
    for b in range(B):
        pc = pc_batch[b].permute(1, 0)               # (3, N)
        centroid = pc.mean(dim=1, keepdim=True)
        pc_centered = pc - centroid
        H = pc_centered @ pc_centered.t() / N
        eigvals, eigvecs = torch.linalg.eigh(H)
        order = torch.argsort(eigvals, descending=True)
        R = eigvecs[:, order]
        pc_rotated = R.t() @ pc_centered
        aligned.append(pc_rotated)
    return torch.stack(aligned, dim=0)                # (B, 3, N)


def emd_align_and_merge(pc_sampled, args):
    """Align chunks via EMD (Hungarian) and concatenate into one tensor per IPC.

    Args:
        pc_sampled: (ipc*samples, npoints, 3)
    Returns:
        (ipc, 3, samples*npoints)
    """
    ipc = args.ipc
    samples = args.samples
    merged_all = []

    for i in range(ipc):
        chunks = pc_sampled[i * samples:(i + 1) * samples]  # (samples, npoints, 3)

        # pad to samples if fewer chunks
        if chunks.shape[0] < samples:
            repeat_times = math.ceil(samples / chunks.shape[0])
            chunks = chunks.repeat(repeat_times, 1, 1)[:samples]

        ref = chunks[0].cpu().numpy()  # (npoints, 3)
        aligned = []

        for j in range(samples):
            tgt = chunks[j].cpu().numpy()          # (npoints, 3)
            cost = cdist(ref, tgt)                 # (npoints, npoints)
            _, col_ind = linear_sum_assignment(cost)
            reordered = tgt[col_ind]               # (npoints, 3)
            aligned.append(torch.from_numpy(reordered))

        merged = torch.cat(aligned, dim=0).unsqueeze(0)  # (1, samples*npoints, 3)
        merged = merged.permute(0, 2, 1)                  # (1, 3, samples*npoints)
        merged_all.append(merged.to(args.device))

    return torch.cat(merged_all, dim=0)                   # (ipc, 3, samples*npoints)


def fps(pc, npoints):
    """Farthest Point Sampling for a single point cloud.

    Args:
        pc: (N, 3) numpy array
        npoints: number of points to sample
    Returns:
        (npoints, 3) numpy array
    """
    N = pc.shape[0]
    sampled = np.zeros(npoints, dtype=int)
    dist = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(npoints):
        sampled[i] = farthest
        centroid = pc[farthest]
        d = np.sum((pc - centroid) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = np.argmax(dist)
    return pc[sampled]


def fps_batch(pc_batch, npoints):
    """Batched FPS using pointops CUDA kernel.

    Args:
        pc_batch: (B, N, 3)
        npoints: int
    Returns:
        (B, npoints, 3)
    """
    idx = pointops.furthestsampling(pc_batch.contiguous(), npoints).long()
    return index_points(pc_batch, idx)


def pc_normalize(pc):
    """Normalize a single point cloud to unit sphere (numpy).

    Args:
        pc: (N, 3) numpy array
    Returns:
        (N, 3) normalized numpy array
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m


def pc_normalize_batch(pc_batch):
    """Normalize a batch of point clouds to unit sphere (torch).

    Args:
        pc_batch: (B, 3, N)
    Returns:
        (B, 3, N) normalized tensor
    """
    centroid = pc_batch.mean(dim=2, keepdim=True)          # (B, 3, 1)
    pc = pc_batch - centroid
    m = pc.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0]  # (B,1,1)
    return pc / (m + 1e-8)


def rotate_pointcloud(pointcloud, axis='y'):
    """Random rotation of a batch of point clouds around the given axis.

    Args:
        pointcloud: (B, 3, N)
        axis: 'x' | 'y' | 'z'
    Returns:
        (B, 3, N) rotated tensor on CUDA
    """
    batch_size = pointcloud.shape[0]
    angles = torch.rand(batch_size) * 2 * np.pi
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    R = torch.eye(3, device=pointcloud.device).unsqueeze(0).repeat(batch_size, 1, 1)
    if axis == 'x':
        R[:, 1, 1] = cos_vals;  R[:, 1, 2] = -sin_vals
        R[:, 2, 1] = sin_vals;  R[:, 2, 2] = cos_vals
    elif axis == 'y':
        R[:, 0, 0] = cos_vals;  R[:, 0, 2] = sin_vals
        R[:, 2, 0] = -sin_vals; R[:, 2, 2] = cos_vals
    elif axis == 'z':
        R[:, 0, 0] = cos_vals;  R[:, 0, 1] = -sin_vals
        R[:, 1, 0] = sin_vals;  R[:, 1, 1] = cos_vals
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    return torch.einsum('bij,bjk->bik', R, pointcloud).cuda()


def voxel_sorting(pc_batch, grid_size=32):
    """Sort points within each cloud by voxel grid index.

    Args:
        pc_batch: (B, 3, N)
        grid_size: voxel resolution
    Returns:
        (B, 3, N) sorted tensor
    """
    B, C, N = pc_batch.shape
    pc = pc_batch.permute(0, 2, 1)  # (B, N, 3)

    min_vals = pc.min(dim=1, keepdim=True)[0]
    max_vals = pc.max(dim=1, keepdim=True)[0]
    scale = (max_vals - min_vals).clamp(min=1e-6)
    pc_norm = (pc - min_vals) / scale  # [0, 1]

    voxel_idx = (pc_norm * (grid_size - 1)).long()  # (B, N, 3)
    flat_idx = (voxel_idx[:, :, 0] * grid_size ** 2
                + voxel_idx[:, :, 1] * grid_size
                + voxel_idx[:, :, 2])              # (B, N)

    sorted_order = flat_idx.argsort(dim=1)         # (B, N)
    pc_sorted = torch.gather(
        pc, 1, sorted_order.unsqueeze(-1).expand(-1, -1, 3)
    )
    return pc_sorted.permute(0, 2, 1)             # (B, 3, N)


def get_uniformity_score(pc):
    """KNN-based uniformity score: std/mean of kNN distances per point.

    Args:
        pc: (B, 3, N)
    Returns:
        (B,) uniformity score per sample
    """
    knn_idx = pointops.knnquery_heap(
        16,
        pc.permute(0, 2, 1).contiguous(),
        pc.permute(0, 2, 1).contiguous()
    ).long()
    knn_xyzs = index_points(pc.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)

    reference = knn_xyzs[:, :, 0:1, :]
    dists = torch.norm(knn_xyzs - reference, dim=-1)        # (B, N, 16)
    mean_dists = dists.mean(dim=2)
    std_dists = dists.std(dim=2)
    uniformity_per_point = std_dists / (mean_dists + 1e-8)
    return uniformity_per_point.mean(dim=-1)                 # (B,)
