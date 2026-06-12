# LADA DPT 官方流程对齐

**日期**: 2026-06-12
**会话概况**: 明确 `main_joint.py` 中 GMM 伪采样实验的原始动机是模拟增量学习中的历史特征不可访问场景，并对照 LADA 论文及官方源码梳理 DPT 的真实用途。

---

## 1. 核心澄清

`debug_lada_eval.py` 中失败的 GMM-LADA 配置并不是 LADA DPT：

- 失败实验使用 GMM 伪特征替代所有真实训练特征，从零构建并微调整个 LADA 分类器。
- 官方 DPT 只为旧任务生成增强原型。
- 当前任务仍使用真实图像特征。
- 旧任务 LADA 参数保持冻结，只更新当前任务新增的 LADA 参数和文本适配器。
- DPT 的目标不是恢复一个可独立训练分类器的完整数据集，而是约束新类决策边界不要侵入旧类区域。

因此，约 3% 的 GMM-LADA 结果不能用于判断官方 DPT 是否有效。

## 2. 官方 LADA/DPT 训练流程

在第 `k` 个任务：

1. 冻结 CLIP 图像编码器。
2. 使用当前任务真实特征做 k-means，初始化当前任务 LADA 单元，默认每类 `lambda1=16`。
3. 从历史 GMM 中每个分量采样一个增强旧类原型：
   `p_tilde = mean + noise * sqrt(Tr(Sigma) / d)`。
4. 拼接旧类增强原型与当前任务真实特征。
5. 分类空间覆盖所有已见类别。
6. 只优化文本适配器和当前任务 LADA 单元；历史 LADA 单元冻结。
7. 任务结束后，用当前任务确定性特征拟合 spherical GMM，默认每类 `lambda2=4`，保存均值、球面方差和混合权重。

论文 Eq. 10 中旧类每个 GMM 分量的 CE 损失由混合权重 `pi` 加权。

## 3. 当前本地实现与官方实现的关键差异

### 3.1 视觉特征空间并非冻结

本地 `main_incremental_lada.py` 默认：

- `tune_vision_encoder=True`
- `lora_type=lora_nsp`

这会让视觉特征空间随任务改变。历史 GMM 参数是在旧特征空间中估计的，但后续训练和测试使用新特征空间，因此旧原型存在表示漂移。

官方 LADA 冻结图像编码器，不存在该问题。

### 3.2 构建原型的数据变换未严格对齐

官方代码使用 `train_loader4updating`：

- 当前 LADA k-means 初始化使用确定性更新 loader
- 任务结束后的 GMM 拟合也使用确定性更新 loader

本地增量入口主要复用带训练增强的数据集构造 `cov_loader`，可能仍执行随机训练变换。此前实验已经证明，随机训练变换会使分类器性能下降约 3-5 个百分点。

### 3.3 特征归一化路径需要以实际张量尺度为准

官方实现：

- LADA k-means 前显式 L2 归一化特征
- DPT GMM 拟合使用其 CLIP 包装模型直接输出
- GMM 增强原型不在采样后手动 L2 归一化

官方包装模型是否已经在输出端归一化需要以源码运行张量为准。对本地 Hugging Face CLIP 而言，LADA 输入和当前训练特征已显式 L2 归一化，因此 GMM 也应在同一单位球面空间拟合。否则旧原型与当前特征尺度不同，指数亲和变换可能失稳。当前复现默认保持 GMM 特征归一化，并保留参数用于后续消融。

### 3.4 本地训练融合加入了额外 mask

官方训练直接将文本 logits 与 LADA logits 相加后计算 CE。

本地 `LADATrainer.train()` 在训练阶段使用：

`mask = (text_preds < lada_classes)`

这属于推理选择逻辑，不应放入官方 DPT 复现实验的训练损失。

### 3.5 默认超参数未对齐

官方主要设置：

- 学习率 `0.001`
- batch size `64`
- LADA 每类单元 `lambda1=16`
- DPT GMM 分量 `lambda2=4`
- 图像编码器冻结

本地默认学习率为 `1e-4`、batch size 为 `32`，并默认同时训练视觉和文本编码器。

## 4. 第一阶段最小实验

先仅验证官方 DPT 语义，不混入 LoRA-NSP：

### 两任务设置

- Task 1: aircraft
- Task 2: caltech101
- 冻结视觉编码器
- 只训练文本适配器和当前 LADA 单元
- 使用确定性 `train_loader4updating`
- `lambda1=16`, `lambda2=4`, `lr=0.001`, `batch_size=64`

### 消融

1. LADA，无旧类回放
2. LADA + 旧类 GMM 均值回放，无噪声
3. LADA + 官方 DPT 增强原型

### 观察指标

- Task 1 训练后准确率
- Task 2 训练后 Task 1 准确率
- Task 2 当前任务准确率
- 两任务全局分类准确率
- 旧类增强原型在训练前后的范数和 cosine 分布

只有该实验复现出 DPT 对旧任务保持的正贡献，才扩展到 10 任务。

## 5. 与 LR-RGDA 的关系

GMM 伪采样仍然适合研究 LR-RGDA 的增量统计回放，但不能直接照搬 LADA DPT 的训练解释：

- LADA DPT 是决策边界约束。
- LR-RGDA 需要的是历史统计重建或分类器参数更新。

二者可以共享轻量 GMM 存储，但训练目标必须分别设计。

## 6. 相关文件与来源

- `src/lada/dpt.py`
- `src/lada/lada_trainer.py`
- `main_incremental_lada.py`
- `debug_lada_eval.py`
- `paper_writing/deep-research-reports/lada/source/2505.23271/example_paper.tex`
- LADA 官方仓库: `https://github.com/MaolinLuo/LADA`
- 官方 `trainer.py`: `https://github.com/MaolinLuo/LADA/blob/main/trainer.py`
- 官方 `models/lada.py`: `https://github.com/MaolinLuo/LADA/blob/main/models/lada.py`
