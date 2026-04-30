import math
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# RFF-based MMD (memory-efficient, for large distributions)
# ---------------------------------------------------------------------------

def rff_chunked_mean(x: torch.Tensor, D: int, sigma: float, chunk_size: int) -> torch.Tensor:
    """Compute RFF mean embedding in chunks to avoid OOM.

    Args:
        x: (N, d) tensor
        D: RFF feature dimension
        sigma: RBF bandwidth
        chunk_size: rows processed at once
    Returns:
        (D,) mean embedding
    """
    N = x.size(0)
    device = x.device
    w = torch.randn(1, D, device=device) / sigma
    b = 2 * math.pi * torch.rand(D, device=device)
    sum_phi = torch.zeros(D, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        proj = x[start:end] @ w + b.unsqueeze(0)
        sum_phi += (math.sqrt(2.0 / D) * torch.cos(proj)).sum(dim=0)
    return sum_phi / N


def compute_mmd_rff_chunked(
    A_flat: torch.Tensor,
    B_flat: torch.Tensor,
    D: int = 256,
    sigma: float = 1.0,
    a_chunk_size: int = 32768,
    b_chunk_size: int = 65536,
) -> torch.Tensor:
    """Approximate MMD² between two scalar distributions using RFF.

    Args:
        A_flat: (N_A,) or (N_A, d)
        B_flat: (N_B,) or (N_B, d)
    Returns:
        scalar MMD²
    """
    A = A_flat.unsqueeze(1) if A_flat.dim() == 1 else A_flat
    B = B_flat.unsqueeze(1) if B_flat.dim() == 1 else B_flat
    mu_A = rff_chunked_mean(A, D, sigma, a_chunk_size)
    mu_B = rff_chunked_mean(B, D, sigma, b_chunk_size)
    diff = mu_A - mu_B
    return (diff * diff).sum()


# ---------------------------------------------------------------------------
# Multi-scale RBF kernel MMD  (M3DLoss — paper's SADM loss)
# ---------------------------------------------------------------------------

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n = L2_distances.shape[0]
            return L2_distances.data.sum() / (n ** 2 - n)
        return self.bandwidth

    def forward(self, X):
        L2 = torch.cdist(X, X) ** 2
        bw = self.get_bandwidth(L2)
        return torch.exp(-L2[None] / (bw * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class M3DLoss(nn.Module):
    """MMD loss with multi-scale RBF kernel (paper's SADM / feature distribution matching)."""

    def __init__(self):
        super().__init__()
        self.kernel = RBF()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        n = X.shape[0]
        XX = K[:n, :n].mean()
        XY = K[:n, n:].mean()
        YY = K[n:, n:].mean()
        return XX - 2 * XY + YY


class RBF_blockwise(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n = L2_distances.shape[0]
            est = L2_distances.data.sum() / (n ** 2 - n)
            if torch.isnan(est) or torch.isinf(est) or est <= 1e-8:
                est = torch.tensor(1e-4, device=L2_distances.device)
            return est
        return self.bandwidth

    def compute_kernel_blockwise(self, X, Y, chunk_size=512):
        N, M = X.shape[0], Y.shape[0]
        bw = self.get_bandwidth(torch.cdist(X, Y).pow(2))
        total, count = 0.0, 0
        for i in range(0, N, chunk_size):
            Xi = X[i:min(i + chunk_size, N)]
            for j in range(0, M, chunk_size):
                Yj = Y[j:min(j + chunk_size, M)]
                L2 = torch.cdist(Xi, Yj).pow(2)
                K = torch.exp(-L2[None] / (bw * self.bandwidth_multipliers)[:, None, None]).sum(0)
                total += K.sum()
                count += Xi.shape[0] * Yj.shape[0]
        return total / count


class M3DLoss_blockwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = RBF_blockwise()

    def forward(self, X, Y):
        return (self.kernel.compute_kernel_blockwise(X, X)
                - 2 * self.kernel.compute_kernel_blockwise(X, Y)
                + self.kernel.compute_kernel_blockwise(Y, Y))


# ---------------------------------------------------------------------------
# Feature matching losses
# ---------------------------------------------------------------------------

def distance_wb(gwr, gws):
    """Cosine distance between gradient tensors (used in match_loss)."""
    shape = gwr.shape
    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)
    dis = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6)
    )
    return dis


def match_loss(gw_syn, gw_real, args):
    """Gradient matching loss (ours / mse / cos)."""
    dis = torch.tensor(0.0).to(args.device)
    if args.dis_metric == 'ours':
        for gwr, gws in zip(gw_real, gw_syn):
            dis += distance_wb(gwr, gws)
    elif args.dis_metric == 'mse':
        real_vec = torch.cat([g.reshape(-1) for g in gw_real])
        syn_vec  = torch.cat([g.reshape(-1) for g in gw_syn])
        dis = torch.sum((syn_vec - real_vec) ** 2)
    elif args.dis_metric == 'cos':
        real_vec = torch.cat([g.reshape(-1) for g in gw_real])
        syn_vec  = torch.cat([g.reshape(-1) for g in gw_syn])
        dis = 1 - torch.sum(real_vec * syn_vec) / (
            torch.norm(real_vec) * torch.norm(syn_vec) + 1e-6
        )
    else:
        raise ValueError(f"unknown dis_metric: {args.dis_metric}")
    return dis


def cosine_similarity_loss(gf, gf_syn):
    """Row-wise cosine similarity loss between two feature matrices."""
    gf_n = gf / (gf.norm(dim=-1, keepdim=True) + 1e-8)
    gf_syn_n = gf_syn / (gf_syn.norm(dim=-1, keepdim=True) + 1e-8)
    return 1 - (gf_n * gf_syn_n).sum(dim=-1).mean()


def cosine_similarity_matrix(gf, gf_syn):
    """Column-wise cosine distance matrix (used as cost for Hungarian matching)."""
    gf_n = gf / (gf.norm(dim=0, keepdim=True) + 1e-8)
    gf_syn_n = gf_syn / (gf_syn.norm(dim=0, keepdim=True) + 1e-8)
    return 1 - (gf_n * gf_syn_n).sum(dim=0)


def optimal_feature_matching(gf, gf_syn):
    """Reorder gf_syn columns to minimise cosine distance via Hungarian algorithm."""
    cost = cosine_similarity_matrix(gf, gf_syn).detach().cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return gf_syn[col_ind]


def pairwise_interpolate_to_length(x, target_length):
    """Linearly interpolate between adjacent points to reach target_length.

    Args:
        x: (B, C, N)
        target_length: int >= N
    Returns:
        (B, C, target_length)
    """
    B, C, N = x.shape
    assert target_length >= N
    num_segments = N - 1
    total_insert = target_length - N
    inserts = [total_insert // num_segments] * num_segments
    for i in range(total_insert % num_segments):
        inserts[i] += 1

    result = []
    for i in range(num_segments):
        p0 = x[:, :, i]
        p1 = x[:, :, i + 1]
        result.append(p0.unsqueeze(-1))
        if inserts[i] > 0:
            alphas = torch.linspace(1, inserts[i], inserts[i], device=x.device) / (inserts[i] + 1)
            interp = ((1 - alphas)[None, None, :] * p0[:, :, None]
                      + alphas[None, None, :] * p1[:, :, None])
            result.append(interp)
    result.append(x[:, :, -1].unsqueeze(-1))
    return torch.cat(result, dim=-1)
