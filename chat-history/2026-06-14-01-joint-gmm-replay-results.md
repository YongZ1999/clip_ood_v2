# Joint GMM 回放实验结果

**日期**: 2026-06-14

## 1. 实验目的

在 `main_joint.py` 框架下，使用完全相同的训练特征公平比较 LADA 和
LR-RGDA，重点验证：

1. 官方 LADA 风格的原始 CLIP embedding 空间 GMM 是否优于球面空间 GMM。
2. 真实特征不可访问时，LADA 和 LR-RGDA 谁更适合使用轻量 GMM 回放。
3. GMM 分量均值回放与按 spherical covariance 添加噪声相比，哪种方式更有效。

## 2. 关键代码修正

### 2.1 修正伪 raw-space GMM

`src/utils/feature_extractor.py::extract_features()` 原本固定执行 L2
归一化。旧版 `main_joint.py` 虽然把其输出保存为 `all_raw_features`，但该张量实际
已经位于单位球面。

本次修改为：

- `extract_features(..., normalize=False)` 显式返回 CLIP visual projection 的原始输出。
- 单位球面特征统一由 raw tensor 使用 `F.normalize` 派生。
- `--gmm_fit_space raw|sphere` 显式控制 GMM 拟合空间。
- GMM 样本在进入 LADA 和 LR-RGDA 前统一 L2 归一化。

因此，2026-05-31 记录的所谓 raw-space GMM 结果实际是 sphere-space 结果，不能作为
官方 raw-space GMM 的实验结论。

### 2.2 消除 LR-RGDA 的真实特征信息泄漏

旧版 `LR-RGDA-GMM` 只用 GMM 样本微调 affine 参数，但解析中心、类协方差和全局
协方差仍由真实特征构建。LADA 则完全由 GMM 伪样本构建，比较不公平。

本次修改后：

- `real`：LADA 和 LR-RGDA 都由同一批真实单位球面特征构建并微调。
- GMM 条件：两者都由同一批归一化 GMM 伪样本构建并微调。
- LR-RGDA 的中心、类协方差、全局协方差也全部从伪样本重新估计。

### 2.3 新增实验工具

- `scripts/run_joint_classifier_replay.sh`
- `scripts/summarize_joint_classifier_replay.py`

运行脚本默认使用 3 个随机种子，并自动生成均值和标准差汇总。

## 3. 服务器运行信息

- 服务器：`raoxuan@10.20.34.30`
- 项目：`/home/raoxuan/projects/project_clip_continual_learning`
- 数据：`/data1/open_datasets/X-TAIL`
- GPU：1
- Conda 环境：`raoxuan`
- 本地提交：`a0a444f`

由于连接服务器 VPN 后无法访问 GitHub，本次没有通过 `git push/pull` 同步，而是使用
`rsync` 将本次涉及的文件直接同步到服务器。服务器上的其他未提交调试文件未改动。

首次启动使用了错误的数据路径 `/home/raoxuan/projects/data/xtail`，在加载 Aircraft
时因缺少 `variants.txt` 退出。定位到历史实验实际使用
`/data1/open_datasets/X-TAIL` 后重新启动，12 组实验均正常完成。

## 4. 实验设置

- X-TAIL 全部 10 个数据集
- 16-shot
- 冻结 CLIP，不执行联合 LoRA 训练：`iterations=0`
- LADA：`k=16`，CE 微调 200 轮
- LR-RGDA：4 centers，rank 32，affine 微调 200 轮
- GMM：每类 4 个 spherical components
- 每类生成 16 个伪特征
- Seeds：42、43、44

四个回放条件：

1. `real`：真实单位球面特征。
2. `gmm_raw`：原始 CLIP embedding 拟合 GMM，按 spherical covariance 采样。
3. `gmm_sphere`：单位球面拟合 GMM，作为旧错误实现对照。
4. `gmm_raw_mean`：原始空间拟合 GMM，只重复分量均值，不添加噪声。

## 5. 最终结果

结果为 3 个随机种子的 ID dataset macro average，格式为 mean +/- std。

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| real | 56.40 +/- 0.00 | 74.71 +/- 0.26 | 74.74 +/- 0.20 | 74.95 +/- 0.26 | 74.74 +/- 0.20 |
| gmm_raw | 56.40 +/- 0.00 | 72.77 +/- 0.18 | 71.78 +/- 0.28 | 73.03 +/- 0.17 | 71.78 +/- 0.28 |
| gmm_sphere | 56.40 +/- 0.00 | 72.85 +/- 0.14 | 71.84 +/- 0.29 | 73.13 +/- 0.18 | 71.84 +/- 0.29 |
| gmm_raw_mean | 56.40 +/- 0.00 | 73.65 +/- 0.19 | 72.52 +/- 0.28 | 73.93 +/- 0.21 | 72.52 +/- 0.28 |

## 6. 当前结论

### 6.1 真实特征条件

LADA 为 `74.74%`，LR-RGDA 为 `74.71%`，差异仅 `0.03` 个百分点。在当前公平实现和
超参数下，两者应视为基本持平，不能声称任何一方稳定优于另一方。

### 6.2 GMM 回放条件

LR-RGDA 在全部 GMM 条件下稳定领先 LADA：

- raw sampling：`+0.99` 个百分点。
- sphere sampling：`+1.01` 个百分点。
- raw mean：`+1.13` 个百分点。

这支持当前项目的核心方向：当真实历史特征不可访问，只能使用轻量分布回放时，
LR-RGDA 比 LADA 更能利用 GMM 伪特征。

### 6.3 Raw 与 Sphere

raw-space GMM 没有优于 sphere-space GMM：

- LR-RGDA：raw `72.77%`，sphere `72.85%`。
- LADA：raw `71.78%`，sphere `71.84%`。

差异小于随机种子波动，当前只能判断二者基本相同。官方源码中的 raw-space 设计并不
意味着在本项目的“用伪样本从零重建整个分类器”设置中必然更优。

### 6.4 均值回放优于加噪采样

`gmm_raw_mean` 显著优于 `gmm_raw`：

- LR-RGDA：`+0.88` 个百分点。
- LADA：`+0.74` 个百分点。

这说明当前 spherical covariance 噪声主要在破坏判别结构，而不是提供有效的数据
增强。对于 16-shot、512 维 CLIP 特征，GMM 分量均值可能已经保留了最可靠的信息。

## 7. 解释边界

该实验在 `main_joint.py` 中把 GMM 伪样本当作真实特征的完整替代，用于从零重建
分类器。它模拟的是“只能保存轻量分布参数”的统计回放场景，但不是官方 LADA DPT
的增量训练语义。

官方 DPT 的旧类 GMM 样本只用于任务增量训练中的边界保持，当前任务仍使用真实特征，
历史 LADA 单元也不会从零重建。因此，本实验可以用于比较 LADA 与 LR-RGDA 的
GMM 特征利用能力，但不能替代 `main_incremental_lada.py` 中的官方 DPT 消融。

## 8. 后续优先事项

1. 把 `gmm_raw_mean` 作为当前最强轻量回放基线。
2. 分析 spherical noise 的尺度，统计采样前后 cosine、类内半径和跨类 margin。
3. 测试方差缩放系数，例如 `0.0, 0.1, 0.25, 0.5, 1.0`，判断最优点是否接近 0。
4. 在增量序列中比较 LADA-DPT 与 LR-RGDA 的历史统计更新，报告平均准确率和遗忘率。

服务器结果目录：

`/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_replay`
