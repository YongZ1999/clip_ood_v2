# LADA 评估指标总结

**日期**: 2026-05-23
**会话概况**: 阅读 LADA 论文（ICML 2025），提取并记录其评估指标定义

---

## 1. 关键讨论 / 决策

- LADA 论文的评估指标沿用自 Zheng et al. (2023) 的 ZSCL 论文
- 三个核心指标：Transfer / Average / Last
- 本项目的实验评估协议与 LADA 一致

## 2. 指标定义

记 $\hat{a}_k^{(j)}$ 为模型在**训练完任务 $j$ 后**，在**任务 $k$** 上的准确率，$K$ 为总任务数。

### Transfer（前向遗忘衡量）

衡量模型保留零样本泛化能力（抗前向遗忘）的程度。

$$\text{Transfer}_k = \frac{1}{k-1}\sum_{j=1}^{k-1} \hat{a}_k^{(j)}, \quad k=2,3,\dots,K$$

最终报告值：$\frac{1}{K-1}\sum_{k=2}^{K} \text{Transfer}_k$（注意 Transfer$_1$ 无定义，因为任务 1 没有"未来任务"）

> ✅ 已通过 LADA 源码 `result_process.py` 验证。代码中对于 K 个任务，transfer_metrics 长度为 K，首列为 "N/A"，其余 K-1 列取均值。

### Average（整体稳定性+可塑性）

所有时间步上任务准确率的平均值，综合反映 stability 和 plasticity。

$$\text{Average}_k = \frac{1}{K}\sum_{j=1}^{K} \hat{a}_k^{(j)}, \quad k=1,2,\dots,K$$

最终报告值：$\frac{1}{K}\sum_{k=1}^{K} \text{Average}_k$（所有 K 列取均值）

> ✅ 已通过 LADA 源码验证。

### Last（后向遗忘衡量）

模型完成所有 $K$ 个任务训练后的最终准确率，衡量 backward forgetting。

$$\text{Last}_k = \hat{a}_k^{(K)}, \quad k=1,2,\dots,K$$

最终报告值：$\frac{1}{K}\sum_{k=1}^{K} \text{Last}_k$（所有 K 列取均值）

> ✅ 已通过 LADA 源码验证。

## 3. 论文实验设置

- **10 个 X-TAIL 数据集**：Aircraft, Caltech101, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397（按字母序）
- **两种设置**：16-shot 和 full-shot
- **基础模型**：CLIP ViT-B/16
- **优化器**：AdamW, lr=0.001, batch size=64

## 4. 源码验证

已通过 LADA 官方 GitHub 仓库 `result_process.py` 确认上述定义。

核心逻辑：
- `result_matrix[j][k]` = 训练完任务 j 后在任务 k 上的准确率（矩阵维度：K×K）
- **Transfer**: 对每列 k 取前 k 行的均值 → 去掉首列 (k=1) → 对剩余 K-1 列取均值
- **Average**: 对每列 k 取所有 K 行的均值 → 对全部 K 列取均值
- **Last**: 对每列 k 取最后一行 values → 对全部 K 列取均值

参考：`https://github.com/MaolinLuo/LADA/blob/main/result_process.py`

## 5. 相关文件

- `paper_writing/reference_papers/25-ICML-LADA Scalable Label-Specific CLIP Adapter for Continual Learning.pdf`: LADA 论文 PDF
- `AGENTS.md`: 项目指南，记录了 LADA 作为核心基线的信息
- `PROJECT_DOCUMENTATION.md`: 项目文档，提到沿用 LADA 的评估协议
