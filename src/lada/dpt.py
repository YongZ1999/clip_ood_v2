import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DPTManager:
    """
    Distribution-Preserved Training (DPT)

    用 GMM 拟合旧类特征分布，训练新任务时采样幻影数据进行蒸馏，
    防止新任务的分类器覆盖旧类的决策边界。

    存储内容：
        - image_prototypes: (N_proto, D) GMM 均值
        - image_prototypes_covs: (N_proto,) 球面协方差
        - image_prototypes_weights: (N_proto,) GMM 混合权重
        - prototype_labels: (N_proto,) 每个原型对应的全局类别 ID
        - text_prototypes: (C_prev, D) 旧类文本特征（冻结）
    """

    def __init__(self, feature_dim: int, prototype_k: int = 4):
        self.feature_dim = feature_dim
        self.prototype_k = prototype_k

        self.image_prototypes = torch.empty(0, feature_dim)
        self.image_prototypes_covs = torch.empty(0)
        self.image_prototypes_weights = torch.empty(0)
        self.prototype_labels = torch.empty(0, dtype=torch.long)
        self.text_prototypes = torch.empty(0, feature_dim)

    def has_prototypes(self):
        """是否有旧任务的原型（第一个任务时返回 False）"""
        return self.image_prototypes.shape[0] > 0

    def update_image_prototypes(self, features, labels, label_offset, k=None):
        """
        训练后更新图像原型

        对当前任务的每个类别拟合 GMM（k 个分量，球面协方差），
        将均值/协方差/权重追加到历史记录。

        Args:
            features: (N, D) 当前任务的图像特征
            labels: (N,) 局部标签（从 0 开始）
            label_offset: 当前任务在全局标签空间的偏移量
            k: GMM 分量数，默认使用 self.prototype_k
        """
        if k is None:
            k = self.prototype_k

        device = features.device
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        unique_labels = np.unique(labels_np)

        all_means = []
        all_covs = []
        all_weights = []
        all_labels = []

        for lbl in unique_labels:
            lbl_indices = np.where(labels_np == lbl)[0]
            lbl_features = features_np[lbl_indices]

            actual_k = min(k, len(lbl_features))
            gmm = GaussianMixture(n_components=actual_k, covariance_type='spherical',
                                  random_state=42)
            gmm.fit(lbl_features)

            means = torch.tensor(gmm.means_, dtype=features.dtype)
            covs = torch.tensor(gmm.covariances_, dtype=features.dtype)
            weights = torch.tensor(gmm.weights_, dtype=features.dtype)

            all_means.append(means)
            all_covs.append(covs)
            all_weights.append(weights)
            all_labels.append(torch.full((actual_k,), lbl + label_offset, dtype=torch.long))

        new_means = torch.cat(all_means, dim=0).to(device)
        new_covs = torch.cat(all_covs, dim=0).to(device)
        new_weights = torch.cat(all_weights, dim=0).to(device)
        new_labels = torch.cat(all_labels, dim=0).to(device)

        self.image_prototypes = torch.cat([self.image_prototypes.to(device), new_means], dim=0)
        self.image_prototypes_covs = torch.cat([self.image_prototypes_covs.to(device), new_covs], dim=0)
        self.image_prototypes_weights = torch.cat([self.image_prototypes_weights.to(device), new_weights], dim=0)
        self.prototype_labels = torch.cat([self.prototype_labels.to(device), new_labels], dim=0)

        logging.info(f"DPT image prototypes updated: {self.image_prototypes.shape[0]} total "
                     f"(+{new_means.shape[0]} new)")

    def update_text_prototypes(self, text_features):
        """
        训练后更新文本原型

        将当前任务的文本特征追加到历史记录。

        Args:
            text_features: (C_curr, D) 当前任务的文本特征（已归一化）
        """
        device = text_features.device
        self.text_prototypes = torch.cat([
            self.text_prototypes.to(device),
            text_features.detach()
        ], dim=0)
        logging.info(f"DPT text prototypes updated: {self.text_prototypes.shape[0]} total")

    def sample_prototypes(self, device, k=1):
        """
        从 GMM 采样幻影数据

        增强公式: p̃ = p + e * sqrt(Tr(Σ) / d), e ~ N(0, I)

        Args:
            device: 目标设备
            k: 每个 GMM 分量采样的数量

        Returns:
            prototypes: (N_proto * k, D) 增强后的原型特征
            labels: (N_proto * k,) 类别标签（全局空间）
            weights: (N_proto * k,) 样本权重（基于 GMM 混合权重）
        """
        if not self.has_prototypes():
            return None, None, None

        means = self.image_prototypes.to(device)
        covs = self.image_prototypes_covs.to(device)
        weights = self.image_prototypes_weights.to(device)
        labels = self.prototype_labels.to(device)

        n_components, n_features = means.shape

        stds = torch.sqrt(covs).unsqueeze(1).repeat(1, k).reshape(-1, 1)
        repeated_means = means.unsqueeze(1).repeat(1, k, 1).reshape(-1, n_features)
        noise = torch.randn(n_components * k, n_features, device=device)
        generated_samples = repeated_means + stds * noise

        weights_per_sample = (weights / k).repeat_interleave(k)
        labels_per_sample = labels.repeat_interleave(k)

        return generated_samples, labels_per_sample, weights_per_sample

    def get_num_text_prototypes(self):
        """返回旧类文本原型的数量"""
        return self.text_prototypes.shape[0]
