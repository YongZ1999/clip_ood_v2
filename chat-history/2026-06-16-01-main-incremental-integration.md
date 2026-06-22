# main_incremental.py 完整集成：训练端 + 推理端

**日期**: 2026-06-16

## 背景

按照 chat-history 中的实验计划，将分开测试的训练端（LoRA-NSP + vision/text + FD/CD + text schedule）和推理端（LR-RGDA 多中心 + 分类器微调 + GMM 统计回放）全部集成到 `main_incremental.py`。

## 已完成集成

### 训练端

- LoRA-NSP + vision/text encoder 双塔微调
- FD/CD 特征/跨模态蒸馏（fd_weight=1.0, cd_weight=1.0）
- Aux 辅助线性分类头（aux_weight=1.0）
- SCE 对称交叉熵（sce_a=0.5, sce_b=0.5）
- text_tuning_schedule（low_lr_after: Task1=1e-4, Task2+=2e-5）
- train_text_encoder 正确传递给 trainer.train()

### 推理端

- 多中心 LR-RGDA（--num_centers, build_multi_center_stats_dict）
- 分类器微调（--rgda_train_iter, --rgda_train_lr）
- GMM 统计回放（--gmm_sample_mode mean, 从 global_stats_dict/center_means 生成旧类伪特征）
- classifier_feature_transform=test（确定性变换）
- 数据集均衡全局协方差
- Ensemble (ZS + LR-RGDA, alpha=0.05)
- LADA 对比参数（--enable_lada 等，已添加 CLI 参数，完整逻辑待接）

### 输出

- JSON 格式兼容 summarize_incremental_metrics.py（accuracy_matrix, args, metrics 标量）
- Transfer/Average/Last 指标（LADA 协议）

## Bug 修复

| Bug | 文件 | 修复 |
|-----|------|------|
| get_full_stats IndexError（三角矩阵）| main_utils.py | matrix[j][k] → matrix[k][j] |
| train_text_encoder 未传递 | main_incremental.py | 添加 train_text_encoder=train_text_this_task |
| main._global_center_means 全局变量泄漏 | main_incremental.py | 改为局部变量 global_center_means |
| dead code global_zs_classifier | main_incremental.py | 删除 |
| docstring alpha=0.5 | main_incremental.py | 改为 0.05 |
| max_num_per_test_dataset 死代码 | main_incremental.py | 改为 eval_max_samples=0 |

## 新增 CLI 参数

- --rgda_train_iter, --rgda_train_lr（分类器微调）
- --gmm_sample_mode, --gaussian_samples_per_class（GMM 回放）
- --enable_lada, --lada_k, --lada_beta, --lada_alpha, --lada_train_iter, --lada_train_lr（LADA 对比）
- --experiment_name, --output_dir（输出）
- --lora_alpha, --lora_dropout, --full_shot, --max_zs_classes（训练）
- --classifier_feature_transform（分类器特征变换）

## 关键文件

- `main_incremental.py` — 增量学习主入口
- `src/trainers/lora_nsp_trainer.py` — 训练器
- `src/utils/main_utils.py` — 评估工具（evaluate_dataset, get_full_stats, print_paper_metrics）
- `src/models/lora_sgp.py` — 双塔 LoRA 模块
- `src/classifiers/gaussian_classifier.py` — LR-RGDA 分类器
- `src/classifiers/lr_rgda_classifier.py` — LR-RGDA fit()
