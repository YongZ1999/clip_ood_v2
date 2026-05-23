import torch
from typing import Dict
from tqdm import tqdm

def cholesky_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    if matrix.dim() == 3:
        batch_size, n, _ = matrix.shape
        reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        reg_eye = reg_eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.linalg.cholesky(matrix + reg_eye)
    
    elif matrix.dim() == 2:
        reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.cholesky(matrix + reg_eye)
    else:
        raise ValueError(f"不支持的矩阵维度: {matrix.dim()}。支持2D或3D张量。")


class GaussianStatistics:
    """Container for per-class Gaussian statistics."""
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-4, cholesky = False):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if mean.dim() != 1:
            raise AssertionError("GaussianStatistics.mean 必须是 1D 向量")

        self.mean = mean
        self.cov = cov
        self.reg = reg

        if cholesky:
            self.L = cholesky_stable(cov, reg=reg)
        else:
            self.L = None

    def to(self, device):
        """Move statistics to the requested device."""

        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        if self.L is not None:
            self.L = self.L.to(device)
        return self

    def sample(
        self,
        n_samples = None,
        cached_eps = None,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""

        if self.L is None:
            self.L = cholesky_stable(self.cov, reg=self.reg)

        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            if n_samples is None:
                raise ValueError("n_samples 必须在未提供 cached_eps 时给定")
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        samples = self.mean.unsqueeze(0) + eps @ self.L.t()
        return samples

class LowRankGaussianStatistics:
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        rank: int = 512,
        reg: float = 1e-8, 
        device=None
    ):
        if mean.dim() != 1:
            raise ValueError("mean must be a 1D vector")
        
        d = mean.size(0)
        if cov.shape != (d, d):
            raise ValueError("cov shape mismatch")

        self.mean = mean
        self.d = d
        self.rank = rank or min(100, d)  # default max rank=100
        self.reg = reg

        U, S, Vh = torch.svd_lowrank(cov, q=rank, niter=4, M=None)
        U = U[:, :self.rank]
        S = S[:self.rank]

        self.U = U
        self.S = torch.clamp(S, min=0.0)

        if device:
            self.to(device)
    @property
    def L(self):
        return self.U * torch.sqrt(self.S).unsqueeze(0)

    @property
    def cov(self) -> torch.Tensor:
        """Reconstruct full covariance (use sparingly for high d!)"""
        return self.L @ self.L.T

    def to(self, device):
        self.mean = self.mean.to(device)
        return self

    def sample(self, n_samples: int) -> torch.Tensor:
        """Efficient sampling without forming full covariance"""
        eps = torch.randn(n_samples, self.L.size(1), device=self.L.device, dtype=self.L.dtype)
        return self.mean.unsqueeze(0) + eps @ self.L.T  # (n_samples, d)


def build_stats_dict_from_features(
    features: torch.Tensor,
    labels: torch.Tensor
) -> Dict[int, 'GaussianStatistics']:
    """
    从特征和标签构建类别统计分布字典

    Args:
        features: 特征向量 [N, D]
        labels: 标签 [N]

    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    unique_classes = torch.unique(labels)
    stats_dict = {}

    for c in unique_classes:
        idx = (labels == c)
        data_c = features[idx]
        mu = torch.mean(data_c, dim=0)
        if data_c.shape[0] > 1:
            cov = torch.cov(data_c.T)
        else:
            cov = torch.eye(data_c.shape[1], device=data_c.device)

        stats_dict[int(c)] = GaussianStatistics(mu, cov)

    return stats_dict


def extract_stats_dict_from_model(
    model,
    dataset_names: list,
    args,
    device: str = "cuda"
) -> Dict[int, 'GaussianStatistics']:
    """
    从数据集和模型提取类别统计分布

    Args:
        model: CLIP模型或其他特征提取器
        dataset_names: 数据集名称列表
        args: 参数对象，需包含root, num_shots等
        device: 计算设备

    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    from utils_data import get_xtail_trainloader, get_transforms

    model.eval()
    stats_dict = {}
    label_offset = 0

    for d_name in dataset_names:
        transform, _ = get_transforms(d_name)
        tr_loader, _, _, c_names = get_xtail_trainloader(
            root=args.root,
            dataset_name=d_name,
            transform_train=transform,
            transform_test=None,
            num_shots=args.num_shots,
            batch_size=32
        )

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in tqdm(tr_loader, desc=f"Extracting {d_name}", leave=False):
                imgs = imgs.to(device)
                feats = model.get_image_features(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
                all_labels.append(lbls + label_offset)

        all_feats = torch.cat(all_feats)
        all_labels = torch.cat(all_labels)

        unique_labels = torch.unique(all_labels)
        for c in unique_labels:
            idx = (all_labels == c)
            data_c = all_feats[idx]
            mu = torch.mean(data_c, dim=0)
            if data_c.shape[0] > 1:
                cov = torch.cov(data_c.T)
            else:
                cov = torch.eye(data_c.shape[1])

            stats_dict[int(c)] = GaussianStatistics(mu, cov)

        label_offset += len(c_names)

    return stats_dict