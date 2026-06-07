import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .da_classifier_builder import LRRGDAClassifierBuilder
from .gaussian_statistics import GaussianStatistics


class LRRGDAClassifier:
    """
    LR-RGDA分类器（基于类别统计分布构建，支持单中心/多中心）

    该类完全基于各类别的高斯统计分布（均值和协方差）构建，
    无需原始数据或特征向量，适用于增量学习场景。

    当 M > 1 时，每类通过 k-means 得到 M 个中心共享协方差，
    通过 log-sum-exp 归约为类概率。

    Usage:
        # 1. 准备类别统计分布
        stats_dict = {
            0: GaussianStatistics(mean_0, cov_0),
            1: GaussianStatistics(mean_1, cov_1),
            ...
        }

        # 2. 构建分类器（单中心）
        classifier = LRRGDAClassifier(stats_dict, device='cuda')

        # 3. 构建分类器（多中心）
        classifier = LRRGDAClassifier(stats_dict, device='cuda', M=16,
                                       center_means=center_means)

        # 4. 预测
        predictions = classifier.predict(features)
    """

    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        device: str = 'cuda',
        rank: int = 32,
        qda_reg_alpha1: float = 0.2,
        qda_reg_alpha2: float = 2.0,
        qda_reg_alpha3: float = 0.5,
        temperature: float = 1.0,
        M: int = 1,
        center_means: Optional[Dict[int, torch.Tensor]] = None,
        global_cov: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            stats_dict: 类别统计分布字典 {class_id: GaussianStatistics}
            device: 计算设备
            rank: 低秩分解的秩
            qda_reg_alpha1: 类内协方差权重
            qda_reg_alpha2: 全局协方差权重
            qda_reg_alpha3: 单位矩阵正则化权重
            temperature: 温度参数（用于概率输出的softmax）
            M: 每类中心数（1=单中心）
            center_means: Dict[class_id, Tensor[M, D]]，M>1 时必填
            global_cov: 外部全局协方差，若提供则每类协方差等权平均被忽略，
                        用于数据集等权平衡
        """
        self.device = device
        self.stats_dict = stats_dict
        self.M = M

        # 构建LR-RGDA分类器
        builder = LRRGDAClassifierBuilder(
            rank=rank,
            qda_reg_alpha1=qda_reg_alpha1,
            qda_reg_alpha2=qda_reg_alpha2,
            qda_reg_alpha3=qda_reg_alpha3,
            temperature=temperature,
            device=device,
            center_means=center_means,
            global_cov=global_cov,
        )

        self.classifier = builder.build(stats_dict)
        self.num_classes = len(stats_dict)
    
    def predict_proba(self, features: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """
        预测概率
        
        Args:
            features: 输入特征 [B, D]
            temperature: 可选的温度参数，覆盖初始化时的设置
            
        Returns:
            概率分布 [B, num_classes]
        """
        return self.classifier.predict_proba(features, temperature=temperature)
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            features: 输入特征 [B, D]
            
        Returns:
            预测类别 [B]
        """
        return self.classifier.predict(features)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播（返回logits）"""
        return self.classifier.forward(features)

    def fit(self, features, labels, iterations=200, lr=0.01, verbose=True):
        """梯度微调 RGDA 分类器权重"""
        self.classifier.fit(features, labels, iterations=iterations, lr=lr, verbose=verbose)
    
    @property
    def class_ids(self):
        """获取类别ID列表"""
        return sorted(self.stats_dict.keys())

class EnsembleClassifier:
    def __init__(self, zeroshot_classifier, lr_rgda_classifier, alpha=0.5, num_id_classes=None,
                 adaptive=False):
        self.zeroshot_classifier = zeroshot_classifier
        self.lr_rgda_classifier = lr_rgda_classifier
        self.alpha = alpha
        self.num_id_classes = num_id_classes
        self.adaptive = adaptive

    def _get_ensemble_logits(self, features, logit_scale_zeroshot):
        """计算融合后的 Logits"""
        zs_logits = (features @ self.zeroshot_classifier) * logit_scale_zeroshot
        rgda_logits = self.lr_rgda_classifier.forward(features)

        if self.adaptive:
            # 逐样本自适应权重：sigmoid(RGDA置信度 - ZS置信度)
            zs_probs = F.softmax(zs_logits, dim=-1)
            rgda_probs = F.softmax(rgda_logits, dim=-1)
            zs_conf = zs_probs.max(dim=-1).values       # [B]
            rgda_conf = rgda_probs.max(dim=-1).values    # [B]
            alpha_sample = torch.sigmoid(rgda_conf - zs_conf).unsqueeze(-1)  # [B, 1]

            if self.num_id_classes is None:
                return (1 - alpha_sample) * zs_logits + alpha_sample * rgda_logits
            else:
                ensemble_logits = (1 - alpha_sample) * zs_logits
                ensemble_logits[:, :self.num_id_classes] += alpha_sample.squeeze(-1) * rgda_logits
                return ensemble_logits
        else:
            # 固定 α
            zs_logits = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
            rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values

            if self.num_id_classes is None:
                return (1 - self.alpha) * zs_logits + self.alpha * rgda_logits
            else:
                ensemble_logits = zs_logits * (1 - self.alpha)
                ensemble_logits[:, :self.num_id_classes] += self.alpha * rgda_logits
                return ensemble_logits

    def predict_proba(self, features, logit_scale_zeroshot):
        """如确实需要概率（比如算置信度）"""
        logits = self._get_ensemble_logits(features, logit_scale_zeroshot)
        return F.softmax(logits, dim=-1)

    def predict(self, features, logit_scale):
        """预测类别标签，直接对 Logits 进行 argmax"""
        logits = self._get_ensemble_logits(features, logit_scale)
        return torch.argmax(logits, dim=1)