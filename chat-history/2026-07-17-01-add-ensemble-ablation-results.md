# 2026-07-17：补充消融实验的 Ensemble 分类器结果

## 需求

将 `docs/paper_experiment_results.md` 中主要只有 Zero-shot（ZS）指标的消融表，补充为同时报告 Ensemble（Ens）分类器的 Transfer / Average / Last。

## 数据来源和范围

- 直接读取 `experiments/paper_formal/E4_lora_nf_hparams/*_ens_results.json`；
- 直接读取 `experiments/paper_formal/E5_distillation/*_ens_results.json`；
- 每个 JSON 的 `metrics.transfer`、`metrics.average`、`metrics.last` 分别写入 Ens 三列，保留两位小数；
- E1、E2、E3、E8 原本已包含 Ens 指标，本次只补齐 E4（NSP 超参数）和 E5（跨模态蒸馏）。

## 重要配置核验

E5 的表头明确规定 `fd_weight=0`。因此 `cd_weight=2.0, cd_temperature=4.0` 不能使用 E1 的默认 LoRA-NF 单次运行（其 `fd_weight=1`）。本次使用配置完全相同的 `E2_components/E2__C2__s43_ens_results.json`：

| 分类器 | Transfer | Average | Last |
|---|---:|---:|---:|
| ZS | 59.70 | 70.00 | 80.09 |
| Ens | 60.02 | 71.86 | 83.73 |

相应地，E5 的结论改为：在 `fd_weight=0` 下，`cd_weight=2.0` 相比 `cd_weight=0` 的 ZS Last 提升 `+0.75`，Ens Last 提升 `+0.54`。这是一处配置一致性更正，不涉及重新训练。

## 解释

- E4 中 `nsp_eps=0.20` 在 ZS Average / Last 上最高；Ens 指标在 `0.05--0.20` 的差异很小，最高 Ens Average 出现在 `0.10`。
- E5 单 seed 结果显示 CD 带来稳定的分类提升；具体峰值随 ZS 或 Ens 指标和权重略有波动，因此论文中应避免把单一 `cd_weight=2.0` 描述为所有指标的唯一最优点。
