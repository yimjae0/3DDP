import math
import torch
import torch.nn as nn


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

