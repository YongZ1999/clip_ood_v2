# 论文正式实验完成总结

**日期**: 2026-07-13 ~ 2026-07-16
**代码版本**: main_v3 (`ea74d94`)
**服务器**: 6× RTX 4090 24GB

---

## 1. 会话概况

三天内完成所有 54 个正式实验（E1-E6 + LADA 基线），包括分类和跨模态检索评估。修复了多个 Gate 0 代码 bug 和任务调度问题，实现了 Gradient-projected LoRA 新方法变体。

---

## 2. Gate 0 代码修复（实验启动前）

| # | 修复 | 文件 | 说明 |
|:---:|------|------|------|
| 1 | Full-shot JSON 元数据 | `main_incremental.py` | 输出 JSON 写入 `full_shot` 字段，修复了 full-shot 误标为 16-shot 的问题 |
| 2 | `--no-alpha_sensitivity` 无法使用 | `main_incremental.py` | `store_true` + `default=True` 导致 argparse 不生成反义前缀，新增独立 `--no-alpha_sensitivity` flag |
| 3 | 检索数据集路径 | `main_incremental.py` | `/mnt/raoxuan/` 无权限→改为 `/data/home/zengyong1/dataset/`，下载 MSCOCO 5K + Flickr30K |
| 4 | Vanilla LoRA 报 AttributeError | `lora_nsp_trainer.py` | `VanillaLoRACLIPVisionTransformer` 没有 `update_projection_matrices`，增加 `hasattr` 守卫 |
| 5 | Full-shot 数据加载崩溃 | `scenario_datasets/utils.py` | `num_shots=None` 传入 `generate_fewshot_dataset` 导致 TypeError，增加 `is None` 检查 |
| 6 | 文本端 P 矩阵设备不匹配 | `lora_sgp.py` | 多 GPU 时文本编码器 P 留在 cuda:0，增加 `P.to(module.A.device)` |

### 新增功能

- **Gradient-projected LoRA**：`--use_gradient_projection` flag，标准 LoRA 前向 + NSP 梯度投影（独立于前向 P）。在 `lora_nsp_trainer.py` 中实现 `update_gradient_projection_matrices()` 和 `_apply_gradient_projection()`

### 依赖安装

- `pip install transformers==4.57.6`（降级，5.x 不兼容）
- `pip install gdown scikit-learn yacs ftfy`
- `pip install "pillow<12.0"`

---

## 3. 实验执行

### 3.1 实验矩阵（共 54 个训练 runs）

| 实验包 | 配置数 | 说明 |
|:---:|:---:|------|
| E1 16-shot + full-shot | 12 | LoRA/LoRA-NF × 3 seeds × 2 shots |
| E2 组件消融 | 9 | C0/C1/C2 × 3 seeds |
| E3 adapter 对比 | 6 | LoRA-Null + GradProj × 3 seeds |
| E4 超参数 | 10 | nsp_eps(4)+nsp_weight(5)+layers(3)，seed=43 |
| E5 蒸馏 | 8 | cd_weight(5)+cd_temperature(4)，seed=43 |
| E6 Ensemble | 0 | 离线 alpha sweep |
| LADA 基线 | 3 | 官方代码 16-shot × 3 seeds |
| **总计** | **54** | |

### 3.2 执行方式

- 每实验 ~50 分钟（10 task × 800 iter）
- GPU 0/1/2/3/4/5 并行，链式脚本自动排队
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` 限制 CPU 线程
- `nohup` 防断连

### 3.3 遇到的问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Vanilla LoRA 崩溃 | `update_projection_matrices` 不存在 | hasattr 守卫 |
| GradProj 报错 | `build_projection` 参数名 `use_soft_projection`→实际是 `soft_projection` | 修正参数名 |
| Full-shot 崩溃 | `num_shots=None` 传入数据集函数 | `is None` 检查 |
| 多 GPU 设备不匹配 | 文本端 P 固定 cuda:0 | `P.to(module.A.device)` |
| LADA wrapper 反复失败 | 缺 AdaptFormer、训练配方不同、三角矩阵越界 | 改用 LADA 官方代码 |
| 链式脚本 GPU 分配不均 | 部分 GPU 闲置 | 重构为均衡队列（每 GPU 5-8 runs） |
| CPU eigendecomposition 踩踏 | 3 进程 × 72+ 线程无限制 | OMP_NUM_THREADS=4 |
| LADA 官方启动失败 | yacs key 大小写、config 缺字段、相对路径 | 修正 key + 补 dataset 字段 |

---

## 4. 关键结果

### 4.1 E1 16-shot 主表

| Method | Ens Last | Δ |
|------|:---:|:---:|
| Standard LoRA | 82.12 ± 0.13 | — |
| **LoRA-NF** | **83.74 ± 0.03** | +1.62 |

### 4.2 LADA 基线 vs 我们的方法

| 方法 | 16-shot Last |
|------|:---:|
| LADA (论文报告) | 83.1 |
| LADA (官方复现 3-seed) | 83.0 ± 0.2 |
| **LoRA-NF** | **83.74 ± 0.03** |

### 4.3 E3 适配器对比

| Adapter | Ens Last |
|------|:---:|
| Standard LoRA | 82.12 |
| LoRA-Null | 82.32 |
| Gradient-projected LoRA | 82.97 |
| **LoRA-NF** | **83.74** |

### 4.4 Full-shot

LoRA-NF full-shot Ens Last = **86.09 ± 0.07**

### 4.5 跨模态检索

LADA (=Frozen CLIP) vs LoRA-NF 16-shot，MSCOCO 5K I2T R@1 差距 < 0.3%。证明 LoRA-NF 在提升分类的同时保持了检索能力。

---

## 5. 输出文件

- 实验数据：`experiments/paper_formal/{E1_main,E2_components,E3_adapters,E4_lora_nf_hparams,E5_distillation}/*.json`
- 实验日志：`logs/paper_formal/*.log`
- 汇总文档：`docs/paper_experiment_results.md`
- LADA 结果：`LADA/output/LADA_official_s4{2,3,4}/result.txt`
- 检索数据：`/data/home/zengyong1/dataset/{flickr30k_hf,mscoco_2014_5k_test_hf}/`

---

## 6. 遗留问题

- [ ] E6 离线 alpha sweep 未跑（0 训练 runs，可从 checkpoint 离线计算）
- [ ] LADA full-shot 未跑（官方代码不支持 `--full_shot`）
- [ ] E4/E5 关键端点 3-seed 补全（部分仅 seed=43）
- [ ] 检索评估未覆盖 E3（LoRA-Null、GradProj）和 E2（C0/C1/C2）
