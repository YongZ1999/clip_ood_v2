# LADA DPT 两任务消融实现

**日期**: 2026-06-12
**会话概况**: 在不改变旧实验默认行为的前提下，为 `main_incremental_lada.py` 增加源码对齐模式和三种旧类回放消融。

---

## 1. 新增实验接口

- `--lada_official_mode`
  - 冻结视觉编码器
  - 使用普通文本 LoRA，避免混入 NSP
  - 使用确定性 `train_loader4updating` 初始化 LADA 和拟合 GMM
  - 训练阶段直接相加 text/LADA logits，不使用推理 mask
  - 跳过未使用的参考数据集加载
- `--lada_replay_mode {none,mean,dpt}`
  - `none`: 不回放旧类
  - `mean`: 只回放 GMM 分量均值
  - `dpt`: 使用论文公式加入球面方差噪声
- `--dpt_feature_normalize`
  - 控制 GMM 拟合空间
  - 当前默认在单位球面特征上拟合，保证旧原型与当前 LADA 输入尺度一致

## 2. 关键实现修正

- `DPTManager.sample_prototypes()` 新增 `add_noise` 参数。
- LADA 初始化可使用独立的确定性 prototype loader。
- 视觉编码器冻结时，训练前向使用 `torch.no_grad()`。
- 文本历史原型与 GMM 回放解耦：即使 `replay_mode=none`，仍保存旧类文本特征以构建全局分类空间。
- 源码对齐模式只改变新实验路径；未启用时保留旧 loader 和训练行为。

## 3. 两任务脚本

新增：

`scripts/run_lada_dpt_ablation.sh`

默认运行：

- aircraft -> caltech101
- 16-shot
- batch size 64
- learning rate 0.001
- 800 iterations
- LADA `lambda1=16`
- GMM `lambda2=4`
- 依次运行 `none`, `mean`, `dpt`

用法：

```bash
bash scripts/run_lada_dpt_ablation.sh 0
```

可通过 `XTAIL_ROOT` 覆盖数据路径。

## 4. 本地验证

- `py_compile` 通过
- `bash -n` 通过
- `git diff --check` 通过
- DPT 均值回放和噪声回放张量单元测试通过
- 本地无法执行完整入口，因为本地环境没有安装 `transformers`；完整实验需在 GPU 服务器运行

## 5. 相关文件

- `src/lada/dpt.py`
- `src/lada/lada_trainer.py`
- `main_incremental_lada.py`
- `scripts/run_lada_dpt_ablation.sh`
