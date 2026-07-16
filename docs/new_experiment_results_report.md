# 新增实验数据报告：SigLIP 2 鲁棒性与检索组件归因

**状态**：E8 已由现有原始 JSON 汇总；E7 待服务器运行。  
**协议说明**：检索为每个任务结束后测得的图文双向 R@K。表中 mR@1 为 I2T R@1 与 T2I R@1 的等权平均；它只用于紧凑展示，不替代原始双向指标。

## E7：SigLIP 2 鲁棒性主表（待运行）

Backbone 固定为 `google/siglip2-base-patch16-224`，X-TAIL 16-shot、seeds 42/43/44。

| Method | Classifier | Transfer | Average | Last |
|---|---|---:|---:|---:|
| LADA-style SigLIP 2 | LADA+ZS | 待运行 | 待运行 | 待运行 |
| LoRA-NF SigLIP 2 | ZS | 待运行 | 待运行 | 待运行 |
| LoRA-NF SigLIP 2 | Ensemble | 待运行 | 待运行 | 待运行 |

> LADA-style SigLIP 2 是架构适配复现（冻结视觉端、文本 AdaptFormer、label-specific prototype memory），不是 OpenAI-CLIP 官方 LADA 代码的直接运行。

## E8.1：FD/CD 组件对检索保持的影响（已有 3-seed 结果）

| Config | FD | CD | MSCOCO Avg mR@1 | MSCOCO Last mR@1 | Flickr Avg mR@1 | Flickr Last mR@1 |
|---|:---:|:---:|---:|---:|---:|---:|
| C0 | 0 | 0 | 42.67 ± 0.09 | **43.39 ± 0.06** | **71.78 ± 0.49** | **72.25 ± 0.64** |
| C1 | 1 | 0 | 42.30 ± 0.10 | 43.35 ± 0.08 | 70.91 ± 0.40 | 71.68 ± 0.85 |
| C2 | 0 | 2 | 42.74 ± 0.08 | 43.25 ± 0.08 | 70.81 ± 0.04 | 71.28 ± 0.18 |
| C3/C4 (full) | 1 | 2 | **42.76 ± 0.10** | 43.24 ± 0.07 | 70.82 ± 0.05 | 71.24 ± 0.21 |

解读：在已有协议下，FD-only（C1）在两套数据集的平均 mR@1 均最低。CD 对 MSCOCO Average 有小幅改善，但 Flickr 的最佳值是无蒸馏 C0；完整配置与 C2 几乎持平。因此这些数据支持“FD 并非检索保持的主要来源”，但**不支持**将任何单一蒸馏组件描述为跨检索集的唯一决定因素。

## E8.2：前向 NSP 对检索保持的影响（已有 3-seed 结果）

| Adapter | MSCOCO Avg mR@1 | MSCOCO Last mR@1 | Flickr Avg mR@1 | Flickr Last mR@1 |
|---|---:|---:|---:|---:|
| Standard LoRA | 42.64 ± 0.05 | 42.78 ± 0.11 | 71.13 ± 0.10 | 71.36 ± 0.12 |
| LoRA-Null | 42.75 ± 0.05 | 43.16 ± 0.05 | **71.17 ± 0.05** | **71.69 ± 0.11** |
| LoRA-NF | **42.76 ± 0.10** | **43.24 ± 0.07** | 70.82 ± 0.05 | 71.24 ± 0.21 |

解读：三种方法均接近 Frozen CLIP 检索水平。LoRA-NF 在 MSCOCO 上有最高的 Average/Last mR@1，LoRA-Null 在 Flickr 上略高；因此可安全表述为“LoRA-NF 没有牺牲跨模态检索”，但不应从这张表单独宣称它在所有检索域严格最优。

## 可复现汇总命令

下列命令生成 E8.1 的 JSON/Markdown，不训练任何模型：

```bash
python scripts/summarize_retrieval_ablation.py \
  --output experiments/paper_formal/E8_retrieval_ablation/E8_components \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s42_retrieval.json \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s43_retrieval.json \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s44_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s42_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s43_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s44_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s42_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s43_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s44_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed42_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed43_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed44_retrieval.json
```

E8.2 使用同一脚本和 `chat-history/2026-07-17-04-server-handoff-siglip2-and-retrieval.md` 中的已核验文件路径；它同样只读取既有 JSON，不训练模型。
