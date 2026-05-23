# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, Optional, List

# 假设 GaussianStatistics 结构如下，为了代码可运行性保留占位
# from classifiers.gaussian_statistics import GaussianStatistics 
class GaussianStatistics:
    def __init__(self, mean, cov, n_samples):
        self.mean = mean
        self.cov = cov
        self.n_samples = n_samples

def get_gpu_memory_info() -> Dict[str, float]:
    """获取当前GPU显存信息"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
    }

class LinearLDAClassifier(nn.Module):
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Optional[Dict[int, float]] = None,
        lda_reg_alpha: float = 0.1,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.lda_reg_alpha = lda_reg_alpha
        self.temperature = temperature
        target_device = torch.device(device)
        
        class_ids = sorted(stats_dict.keys())
        self.num_classes = len(class_ids)

        # 使用 no_grad 避免构建计算图，大幅节省显存
        with torch.no_grad():
            # === Step 1. 快速构建均值矩阵和全局协方差 ===
            # 预先读取第一个元素确定维度
            first_stats = stats_dict[class_ids[0]]
            d = first_stats.cov.size(0)
            
            # 使用列表推导式快速收集数据
            means_list = [stats_dict[cid].mean.to(target_device) for cid in class_ids]
            means = torch.stack(means_list) # [C, D]

            # 流式累加协方差，避免同时加载所有 Cov 矩阵
            global_cov = torch.zeros(d, d, device=target_device)
            for cid in class_ids:
                global_cov.add_(stats_dict[cid].cov.to(target_device))
            
            global_cov.div_(self.num_classes)

            # === Step 2. Spherical 正则化 ===
            # In-place 操作节省显存
            global_cov.mul_(1.0 - self.lda_reg_alpha)
            global_cov.diagonal().add_(self.lda_reg_alpha)

            # === Step 3. 计算逆矩阵 (Cholesky 优先) ===
            # 添加微小抖动保证正定
            global_cov.diagonal().add_(1e-6)
            
            try:
                L = torch.linalg.cholesky(global_cov)
                cov_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                logging.warning("LDA Covariance not positive definite, falling back to pinv.")
                cov_inv = torch.linalg.pinv(global_cov)

            # === Step 4. 向量化计算权重 & 偏置 ===
            # 准备先验
            if class_priors is None:
                log_priors = torch.full((self.num_classes,), -math.log(self.num_classes), device=target_device)
            else:
                priors = [class_priors[cid] for cid in class_ids]
                log_priors = torch.tensor(priors, device=target_device).log()

            # 向量化计算: W = Σ^{-1} μ^T -> [D, C]
            # w_c = cov_inv @ mu
            W = cov_inv @ means.T  # [D, D] @ [D, C] -> [D, C]
            
            # 向量化计算: b = -0.5 * μ^T Σ^{-1} μ + log(π)
            # 技巧: diag(μ @ W) 等价于 sum(μ * W.T, dim=1)
            # means: [C, D], W.T: [C, D]
            mahalanobis_term = -0.5 * (means * W.T).sum(dim=1) # [C]
            b = mahalanobis_term + log_priors

            # === Step 5. 线性层承载 ===
            self.linear = nn.Linear(d, self.num_classes, bias=True)
            self.linear.weight.data.copy_(W.T) # Linear 权重存储为 [Out, In] 即 [C, D]
            self.linear.bias.data.copy_(b)
            self.linear.requires_grad_(False)
            
            # 将模型移动到指定设备
            self.to(target_device)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)


class RegularizedGaussianDA(nn.Module):
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Dict[int, float] = None,
        qda_reg_alpha1: float = 0.2,
        qda_reg_alpha2: float = 2.0,
        qda_reg_alpha3: float = 0.5,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        target_device = torch.device(device)
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.epsilon = 1e-4  # 增加一点以提高稳定性
        self.temperature = temperature

        with torch.no_grad():
            # 1. 准备先验
            if class_priors is None:
                log_priors_t = torch.full((self.num_classes,), -math.log(self.num_classes), device=target_device)
            else:
                priors_list = [class_priors[cid] for cid in self.class_ids]
                log_priors_t = torch.tensor(priors_list, device=target_device).log()
            
            self.register_buffer("log_priors", log_priors_t)

            # 2. 收集均值 (Covariance 稍后处理以节省瞬时内存)
            means_list = [stats_dict[cid].mean.float().to(target_device) for cid in self.class_ids]
            means = torch.stack(means_list) # [C, D]
            self.register_buffer("means", means)
            
            D = means.shape[1]
            
            # 3. 计算全局协方差 (用于收缩)
            global_cov = torch.zeros(D, D, device=target_device)
            for cid in self.class_ids:
                global_cov.add_(stats_dict[cid].cov.float().to(target_device))
            global_cov.div_(self.num_classes)
            # 对称化
            global_cov = 0.5 * (global_cov + global_cov.T)

            # 4. 逐个处理协方差矩阵 (避免创建 [C, D, D] 的巨大张量)
            cov_invs_list = []
            logdets_list = []
            
            identity = torch.eye(D, device=target_device)
            
            for cid in self.class_ids:
                # 获取原始协方差
                sigma = stats_dict[cid].cov.float().to(target_device)
                sigma = 0.5 * (sigma + sigma.T) # 确保对称
                
                # 正则化混合: α1*Σ + α2*Σ_global + α3*I
                reg_sigma = (qda_reg_alpha1 * sigma) + \
                            (qda_reg_alpha2 * global_cov) + \
                            (qda_reg_alpha3 * identity)
                
                # 额外的数值稳定性处理
                reg_sigma.diagonal().add_(self.epsilon)
                
                # Cholesky 分解与求逆
                # 使用 linalg.cholesky_ex 可以在失败时捕获错误而不崩溃
                L, info = torch.linalg.cholesky_ex(reg_sigma)
                
                if info.item() == 0:
                    # 成功
                    inv = torch.cholesky_inverse(L)
                    logdet = 2 * L.diagonal().log().sum()
                else:
                    # 失败，回退到 Eigendecomposition (比 SVD 对对称矩阵更快)
                    logging.warning(f"Cholesky failed for class {cid}, using eigh.")
                    evals, evecs = torch.linalg.eigh(reg_sigma)
                    evals = torch.clamp(evals, min=1e-6)
                    inv = evecs @ torch.diag(1.0 / evals) @ evecs.T
                    logdet = evals.log().sum()

                cov_invs_list.append(inv)
                logdets_list.append(logdet)

            # 最终堆叠
            self.register_buffer("cov_invs", torch.stack(cov_invs_list)) # [C, D, D]
            self.register_buffer("logdets", torch.stack(logdets_list))   # [C]

        logging.info(f"[RegularizedGaussianDA] Init done. Shape: {self.cov_invs.shape}")

    @property
    def device(self) -> torch.device:
        return self.means.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        计算 logits。对于大 Batch 或大维度，自动进行分块计算以防 OOM。
        """
        B, D = x.shape
        C = self.num_classes
        
        # 显存估算: 需要创建差分张量 [B, C, D]
        # 如果 B*C*D * 4bytes > 1GB (举例), 则应该分块
        # 比如 B=64, C=1000, D=768 -> 64*1000*768*4 / 1e9 ~= 0.2GB (安全)
        # 比如 B=128, C=2000, D=4096 -> 4GB (危险)
        
        # 简单策略：如果 B*C > 20000 且 D > 1024，则对 Batch 分块
        CHUNK_SIZE = 32
        if B * C > 50000: 
            logits_list = []
            for i in range(0, B, CHUNK_SIZE):
                x_chunk = x[i:i+CHUNK_SIZE]
                logits_list.append(self._forward_chunk(x_chunk))
            return torch.cat(logits_list, dim=0)
        else:
            return self._forward_chunk(x)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # 扩展维度进行广播: [B, 1, D] - [1, C, D] -> [B, C, D]
        # 注意: 如果 D 很大，这里是显存瓶颈
        xc = x.unsqueeze(1) - self.means.unsqueeze(0) 
        
        # 计算 Mahalanobis: (x-u)^T Σ^-1 (x-u)
        # v = Σ^-1 (x-u) : [C, D, D] @ [B, C, D, 1] -> [B, C, D] 
        # 但 einsum 更简洁: 'bcd, cde -> bce' (batch, class, dim_in), (class, dim_in, dim_out)
        v = torch.einsum("bcd,cde->bce", xc, self.cov_invs)
        
        # 点积: [B, C, D] * [B, C, D] -> sum -> [B, C]
        maha = 0.5 * (v * xc).sum(dim=-1)
        
        # Logits: -0.5 * maha - 0.5 * logdet + log_prior
        logits = -maha - 0.5 * self.logdets.unsqueeze(0) + self.log_priors.unsqueeze(0)
        return logits

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor, temperature: Optional[float] = None):
        """
        预测概率，支持温度缩放
        
        Args:
            x: 输入特征 [B, D]
            temperature: 温度系数，如果为None则使用初始化时的temperature值
        """
        logits = self.forward(x)
        temp = temperature if temperature is not None else self.temperature
        return F.softmax(logits / temp, dim=1)


class LRRGDA(nn.Module):
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        rank: int = 64,
        class_priors = None,
        qda_reg_alpha1: float = 0.2,
        qda_reg_alpha2: float = 2.0,
        qda_reg_alpha3: float = 0.5,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 20,
        center_means: Optional[Dict[int, torch.Tensor]] = None,
    ):
        super().__init__()
        target_device = torch.device(device)

        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.rank = rank
        self.temperature = temperature
        self.num_centers = 1  # 默认单中心

        # 检查是否使用多中心
        use_multi_center = center_means is not None
        if use_multi_center:
            sample_centers = center_means[self.class_ids[0]]
            M = sample_centers.shape[0]
            self.num_centers = M
            logging.info(f"[Multi-Center] M={M}, total pseudo-classes={self.num_classes * M}")

        # 注册用于校验的 buffer
        sample_stats = stats_dict[self.class_ids[0]]
        D = sample_stats.mean.shape[0]
        self.register_buffer("_feature_dim", torch.tensor(D))
        self.register_buffer("_rank", torch.tensor(rank))

        # 使用 no_grad 块，这是优化 LowRank 初始化内存的关键
        with torch.no_grad():
            logging.info(f"[Init] Starting batched LRRGDA on {device}. D={D}, Rank={rank}")

            # === 1. 计算全局协方差 (Basis Matrix A) ===
            global_cov = torch.zeros((D, D), device=target_device)

            # 分批读取
            all_cids = self.class_ids
            for i in range(0, self.num_classes, batch_size):
                batch_cids = all_cids[i:i + batch_size]
                for cid in batch_cids:
                    s = stats_dict[cid]
                    global_cov.add_(s.cov.to(target_device).float())

            global_cov.div_(self.num_classes)

            # === 2. 计算基矩阵 A 的逆 ===
            # A = α2 * Σ_global + α3 * I
            A = qda_reg_alpha2 * global_cov
            A.diagonal().add_(qda_reg_alpha3)

            L_A, info = torch.linalg.cholesky_ex(A)
            if info.item() == 0:
                A_inv = torch.cholesky_inverse(L_A)
                base_logdet = 2 * L_A.diagonal().log().sum()
            else:
                logging.warning("Global Cov A is singular, using inv.")
                A_inv = torch.linalg.inv(A)
                base_logdet = torch.logdet(A)

            # === 3. 计算低秩部分 U 和 Woodbury 修正项 ===
            # 缓存列表
            cls_bias_list = []  # 类共享偏置 [-0.5*log|Σ| + log(π)]
            U_eff_T_B_inv_list = []
            M_inv_list = []

            # 准备先验
            if class_priors is None:
                log_priors = torch.full((self.num_classes,), -math.log(self.num_classes), device=target_device)
            else:
                priors_list = [class_priors[cid] for cid in self.class_ids]
                log_priors = torch.tensor(priors_list, device=target_device).log()

            if use_multi_center:
                # ========== 多中心模式 ==========
                w_mc_list = []    # [C, M, D]
                b_mc_list = []    # [C, M]（仅 Mahalanobis 常数，不含 logdet/prior）
                z_j_list = []     # [C, M, r]
                quad_const_list = []  # [C, M]

                for i in range(0, self.num_classes, batch_size):
                    batch_indices = slice(i, i + batch_size)
                    batch_cids = self.class_ids[batch_indices]
                    current_batch_size = len(batch_cids)

                    # 收集当前批次的 Cov
                    batch_covs = []
                    for cid in batch_cids:
                        batch_covs.append(stats_dict[cid].cov.to(target_device).float())
                    batch_covs = torch.stack(batch_covs)  # [B_size, D, D]

                    # SVD (shared across centers)
                    U_batch, S_batch, _ = torch.svd_lowrank(batch_covs, q=self.rank, niter=2)
                    S_batch = torch.clamp(S_batch, min=1e-7)

                    scale = torch.sqrt(qda_reg_alpha1 * S_batch)
                    U_eff = U_batch * scale.unsqueeze(1)  # [B_size, D, r]

                    # Woodbury M
                    Ai_U = A_inv @ U_eff
                    M_batch = U_eff.transpose(1, 2) @ Ai_U
                    M_batch.diagonal(dim1=-2, dim2=-1).add_(1.0)

                    L_M, info_M = torch.linalg.cholesky_ex(M_batch)
                    is_pd = (info_M == 0)
                    M_inv_batch = torch.zeros_like(M_batch)
                    logdet_batch = torch.zeros(current_batch_size, device=target_device)

                    if is_pd.all():
                        M_inv_batch = torch.cholesky_inverse(L_M)
                        logdet_batch = 2 * L_M.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
                    else:
                        for b_idx in range(current_batch_size):
                            if is_pd[b_idx]:
                                M_inv_batch[b_idx] = torch.cholesky_inverse(L_M[b_idx])
                                logdet_batch[b_idx] = 2 * L_M[b_idx].diagonal().log().sum()
                            else:
                                M_inv_batch[b_idx] = torch.linalg.inv(M_batch[b_idx])
                                logdet_batch[b_idx] = torch.logdet(M_batch[b_idx])

                    total_logdet = base_logdet + logdet_batch
                    cls_bias = -0.5 * total_logdet + log_priors[batch_indices]  # [B_size]

                    # U_eff_T_B_inv: U^T A^{-1} [B_size, r, D] → 类内共享
                    U_eff_T_B_inv_batch = U_eff.transpose(1, 2) @ A_inv

                    # 逐中心计算
                    w_center = []
                    b_center = []
                    z_center = []
                    qconst_center = []

                    for b_idx, cid in enumerate(batch_cids):
                        centers = center_means[cid].to(target_device).float()  # [M, D]
                        M_c = centers.shape[0]

                        # w_j = A^{-1} μ_j
                        w_j = centers @ A_inv  # [M, D]

                        # b_j (纯 Mahalanobis 常数，不含 logdet/prior)
                        b_j = -0.5 * (centers * w_j).sum(dim=1)  # [M]

                        # z_j = U^T A^{-1} μ_j
                        U_inv = U_eff_T_B_inv_batch[b_idx]  # [r, D]
                        z_j = (U_inv @ centers.T).T  # [M, r]

                        # quad_const_j = 0.5 * z_j^T M^{-1} z_j
                        M_inv_c = M_inv_batch[b_idx]  # [r, r]
                        M_z_j = z_j @ M_inv_c  # [M, r]
                        qconst_j = 0.5 * (z_j * M_z_j).sum(dim=1)  # [M]

                        w_center.append(w_j)
                        b_center.append(b_j)
                        z_center.append(z_j)
                        qconst_center.append(qconst_j)

                    # 组装 batch 结果
                    w_mc_list.append(torch.stack(w_center, dim=0))        # [B_size, M, D]
                    b_mc_list.append(torch.stack(b_center, dim=0))        # [B_size, M]
                    z_j_list.append(torch.stack(z_center, dim=0))         # [B_size, M, r]
                    quad_const_list.append(torch.stack(qconst_center, dim=0))  # [B_size, M]
                    cls_bias_list.append(cls_bias)                        # [B_size]
                    U_eff_T_B_inv_list.append(U_eff_T_B_inv_batch)        # [B_size, r, D]
                    M_inv_list.append(M_inv_batch)                        # [B_size, r, r]

                    del U_batch, S_batch, U_eff, Ai_U, M_batch, L_M

                # 注册 buffer
                self.register_buffer("affine_weights", torch.cat(w_mc_list, dim=0))        # [C, M, D]
                self.register_buffer("affine_biases", torch.cat(b_mc_list, dim=0))         # [C, M]
                self.register_buffer("z_j", torch.cat(z_j_list, dim=0))                     # [C, M, r]
                self.register_buffer("quad_consts", torch.cat(quad_const_list, dim=0))      # [C, M]
                self.register_buffer("cls_biases", torch.cat(cls_bias_list, dim=0))         # [C]
                self.register_buffer("U_eff_T_B_inv", torch.cat(U_eff_T_B_inv_list, dim=0)) # [C, r, D]
                self.register_buffer("M_invs", torch.cat(M_inv_list, dim=0))                # [C, r, r]

            else:
                # ========== 单中心模式（与原逻辑一致）==========
                w_c_list = []
                b_c_list = []
                U_eff_T_B_inv_list = []
                U_eff_T_B_inv_mu_list = []
                M_inv_list = []

                for i in range(0, self.num_classes, batch_size):
                    batch_indices = slice(i, i + batch_size)
                    batch_cids = self.class_ids[batch_indices]
                    current_batch_size = len(batch_cids)

                    batch_covs = []
                    for cid in batch_cids:
                        batch_covs.append(stats_dict[cid].cov.to(target_device).float())
                    batch_covs = torch.stack(batch_covs)

                    U_batch, S_batch, _ = torch.svd_lowrank(batch_covs, q=self.rank, niter=2)
                    S_batch = torch.clamp(S_batch, min=1e-7)

                    scale = torch.sqrt(qda_reg_alpha1 * S_batch)
                    U_eff = U_batch * scale.unsqueeze(1)

                    Ai_U = A_inv @ U_eff
                    M_batch = U_eff.transpose(1, 2) @ Ai_U
                    M_batch.diagonal(dim1=-2, dim2=-1).add_(1.0)

                    L_M, info_M = torch.linalg.cholesky_ex(M_batch)
                    is_pd = (info_M == 0)
                    M_inv_batch = torch.zeros_like(M_batch)
                    logdet_batch = torch.zeros(current_batch_size, device=target_device)

                    if is_pd.all():
                        M_inv_batch = torch.cholesky_inverse(L_M)
                        logdet_batch = 2 * L_M.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
                    else:
                        for b_idx in range(current_batch_size):
                            if is_pd[b_idx]:
                                M_inv_batch[b_idx] = torch.cholesky_inverse(L_M[b_idx])
                                logdet_batch[b_idx] = 2 * L_M[b_idx].diagonal().log().sum()
                            else:
                                M_inv_batch[b_idx] = torch.linalg.inv(M_batch[b_idx])
                                logdet_batch[b_idx] = torch.logdet(M_batch[b_idx])

                    batch_means = stats_dict[batch_cids[0]].mean.new_zeros(current_batch_size, D)
                    for b_idx, cid in enumerate(batch_cids):
                        batch_means[b_idx] = stats_dict[cid].mean.to(target_device).float()

                    w_c_batch = batch_means @ A_inv
                    maha_const = -0.5 * (batch_means * w_c_batch).sum(dim=1)
                    total_logdet = base_logdet + logdet_batch
                    b_c_batch = maha_const - 0.5 * total_logdet + log_priors[batch_indices]

                    U_eff_T_B_inv_batch = U_eff.transpose(1, 2) @ A_inv
                    U_eff_T_B_inv_mu_batch = (U_eff_T_B_inv_batch @ batch_means.unsqueeze(-1)).squeeze(-1)

                    w_c_list.append(w_c_batch)
                    b_c_list.append(b_c_batch)
                    U_eff_T_B_inv_list.append(U_eff_T_B_inv_batch)
                    U_eff_T_B_inv_mu_list.append(U_eff_T_B_inv_mu_batch)
                    M_inv_list.append(M_inv_batch)

                    del U_batch, S_batch, U_eff, Ai_U, M_batch, L_M

                self.register_buffer("affine_weights", torch.cat(w_c_list, dim=0))          # [C, D]
                self.register_buffer("affine_biases", torch.cat(b_c_list, dim=0))           # [C]
                self.register_buffer("U_eff_T_B_inv", torch.cat(U_eff_T_B_inv_list, dim=0)) # [C, r, D]
                self.register_buffer("U_eff_T_B_inv_mu", torch.cat(U_eff_T_B_inv_mu_list, dim=0)) # [C, r]
                self.register_buffer("M_invs", torch.cat(M_inv_list, dim=0))                # [C, r, r]

        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> torch.device:
        return self.affine_weights.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_centers == 1:
            # ========== 单中心 ==========
            affine_logits = F.linear(x, self.affine_weights, self.affine_biases)
            U_term = torch.einsum('crd,bd->bcr', self.U_eff_T_B_inv, x)
            u_c = U_term - self.U_eff_T_B_inv_mu.unsqueeze(0)
            M_u = torch.einsum('bcr,crk->bck', u_c, self.M_invs)
            quadratic = 0.5 * (u_c * M_u).sum(dim=-1)
            return affine_logits + quadratic

        else:
            # ========== 多中心: log-sum-exp over M centers ==========
            B, C, M = x.shape[0], self.num_classes, self.num_centers

            # 1. 仿射部分 [B, C, M]
            # affine_weights: [C, M, D], affine_biases: [C, M]
            # x: [B, D] → x @ w^T: [B, D] @ [C*M, D]^T = [B, C*M]
            w_flat = self.affine_weights.reshape(-1, self.affine_weights.shape[-1])  # [C*M, D]
            b_flat = self.affine_biases.reshape(-1)  # [C*M]
            affine_flat = F.linear(x, w_flat, b_flat)  # [B, C*M]
            affine = affine_flat.reshape(B, C, M)  # [B, C, M]

            # 2. 二次部分: 利用共享分解 quad_j = quad_base - cross_j + quad_const_j
            # z = U^T A^{-1} x [B, C, r] — 类内共享
            z = torch.einsum('crd,bd->bcr', self.U_eff_T_B_inv, x)
            M_inv_z = torch.einsum('bcr,crk->bck', z, self.M_invs)  # [B, C, r]
            quad_base = 0.5 * (z * M_inv_z).sum(dim=-1)  # [B, C]

            # cross_j = z^T M^{-1} z_j = (M_inv_z)^T z_j  [B, C, M]
            cross = torch.einsum('bcr,cmr->bcm', M_inv_z, self.z_j)  # [B, C, M]

            # quad per center: [B, C, M]
            quad = quad_base.unsqueeze(-1) - cross + self.quad_consts.unsqueeze(0)

            # 3. 合并并 + 类共享偏置
            logits = affine + quad  # [B, C, M]

            # 4. log-sum-exp over M
            m = logits.max(dim=-1, keepdim=True).values
            per_center_exp = torch.exp(logits - m)
            logsum = m.squeeze(-1) + torch.log(per_center_exp.sum(dim=-1))  # [B, C]

            # 5. 加上类共享偏置（logdet + prior）
            return logsum + self.cls_biases.unsqueeze(0)

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor, temperature: Optional[float] = None):
        """
        预测概率，支持温度缩放
        
        Args:
            x: 输入特征 [B, D]
            temperature: 温度系数，如果为None则使用初始化时的temperature值
        """
        logits = self.forward(x)
        temp = temperature if temperature is not None else self.temperature
        return F.softmax(logits / temp, dim=1)
