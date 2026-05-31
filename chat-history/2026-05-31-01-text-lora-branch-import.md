# 引入 YongZ v2-text-lora 分支并测试

**日期**: 2026-05-31
**会话概况**: 分析 YongZ1999/CLIP_OOD 仓库的 v2-text-lora 分支与当前代码的差异，将其完整移植为本地 `v2-text-lora` 分支以便后续测试，再决定是否合并回 main。

---

## 1. 关键讨论 / 决策

- **决策 1: 走"独立分支"而非直接合并主线**
  - 先把 v2 改动落到 `v2-text-lora` 分支验证有效性
  - 待跑出实验对比（Last / Avg / Transfer）后再决定合不合 main
  - 避免一次性引入太大改动污染 main 的稳定性

- **决策 2: v2 内容的接入方式选"拷贝到新本地分支"**
  - 不添加 YongZ 为 remote，本地仓库自包含
  - 从 main 拉新分支后，把 v2 commit (`31cddb7`) 涉及的 7 个文件直接覆盖、单 commit 提交
  - commit message 标注来源（YongZ commit hash）

- **决策 3: 切分支前先把 `scripts/run_tier1.sh` 合入 main**
  - 防止未跟踪文件在 v2 分支上造成混淆
  - 该脚本是 Tier 1 投影/初始化方法对比的运行脚本，本就属于实验基础设施

---

## 2. 重要发现

### 2.1 v2-text-lora 分支的"净改动"
剥掉更早 commit（SCE、kmeans 多中心、aux head 等）后，本次 v2 commit `31cddb7` 实际只做两件事：

1. **文本侧 LoRA-NSP**：
   - `src/models/lora_sgp.py` 新增 `LoRACLIPTextTransformer`（与视觉版对称的类，约 100 行）
   - `src/models/clip.py`：`tune_text_encoder=True` 时给 `model.text_model` 包一层
   - `src/trainers/lora_nsp_trainer.py`：双编码器协方差/投影维护，训练循环里 ZS 分类器每步重算并保留梯度（让文本编码器走 CE）；联合训练加 `max_zs_classes` 子采样
   - `main_incremental.py` 默认 `--tune_text_encoder True`；`main_joint.py` 默认 False（1100 类显存压力大）
   - checkpoint 加 `text_covariance_history`

2. **Transfer 指标 bugfix**（`src/utils/main_utils.py::get_full_stats`）
   - `sum(matrix[k][j] for j in range(k))` → `sum(matrix[j][k] for j in range(k))`
   - 原来沿行错误聚合，正确应当沿列：任务 k 在之前 j 个任务上的迁移表现

### 2.2 代码结构可改进的点（未来重构方向）
- **视觉/文本 LoRA 类大量复制**：可抽 `_LoRACLIPEncoderBase` 基类
- **Trainer 双轨维护**：可改成 dict-of-dicts `covariance_histories = {"vision": {}, "text": {}}`，extract/update 共用一套
- **ZS 分类器构建**：抽 `ZeroShotClassifierBuilder` 把"每步重算 + 子采样"封装起来
- **text covariance 提取的 attention_mask**：v2 用 `(input_ids != 0).long()`，但 CLIPTokenizer 的 pad token id 是 `49407` 而非 0，应改用 processor 输出的 mask（潜在 bug）
- **Transfer bugfix 应归到 `src/utils/metrics.py`** 而非 `main_utils.py`

这些重构留待 v2 验证有效后、合并到 main 时再做。

---

## 3. 操作记录

```bash
# 1. main 上提交 run_tier1.sh
git add scripts/run_tier1.sh
git commit -m "[实验] 添加 tier1 投影/初始化方法对比脚本"
# → 2711bdf

# 2. 从 main 拉分支
git checkout -b v2-text-lora

# 3. 把 /tmp/CLIP_OOD_v2 (clone 自 YongZ1999/CLIP_OOD@v2-text-lora) 的 7 个文件覆盖
cp /tmp/CLIP_OOD_v2/main_incremental.py main_incremental.py
cp /tmp/CLIP_OOD_v2/main_joint.py main_joint.py
cp /tmp/CLIP_OOD_v2/src/models/clip.py src/models/clip.py
cp /tmp/CLIP_OOD_v2/src/models/lora_sgp.py src/models/lora_sgp.py
cp /tmp/CLIP_OOD_v2/src/trainers/lora_nsp_trainer.py src/trainers/lora_nsp_trainer.py
cp /tmp/CLIP_OOD_v2/src/utils/main_utils.py src/utils/main_utils.py
cp /tmp/CLIP_OOD_v2/.gitignore .gitignore

# 4. 单 commit 提交
git add -A
git commit -m "v2: text LoRA-NSP + Transfer bugfix"
# → d4242eb

# 5. 6 个 Python 文件 AST 语法检查全部通过
```

**diff stat**:
```
 .gitignore                       |   9 +-
 main_incremental.py              |  25 ++
 main_joint.py                    |   6 +
 src/models/clip.py               |  24 +-
 src/models/lora_sgp.py           | 101 ++++++-
 src/trainers/lora_nsp_trainer.py | 588 ++++++++++++++++++++-------------------
 src/utils/main_utils.py          |  36 ++-
 7 files changed, 485 insertions(+), 304 deletions(-)
```

与上游 commit `31cddb7` 完全一致。

---

## 4. 待办事项 / 遗留问题

- [ ] **push v2-text-lora 到远程**，方便服务器 git pull
- [ ] **服务器端实验对比**：
  - 短序列（aircraft / caltech101 / dtd）跑 `--tune_text_encoder True` vs `False`
  - 看 Last / Avg / Transfer 三个指标变化
  - 同时确认 Transfer bugfix 的数值影响
- [ ] **联合训练 (`main_joint.py`) 显存测试**：开启 `--tune_text_encoder True` 时 1100 类的显存峰值，确认 `max_zs_classes` 默认值（128）是否合理
- [ ] **决定是否 merge v2-text-lora 到 main**：
  - 验证有效 → 重构（按上面 §2.2 4 点）→ merge
  - 验证无效 → 保留分支留档，回 main 继续
- [ ] **潜在 bug 排查**：text covariance 提取里 attention_mask 用 `(input_ids != 0).long()` 是否正确

---

## 5. 相关文件

| 文件 | 改动 |
|------|------|
| `src/models/lora_sgp.py` | +101 行；新增 `LoRACLIPTextTransformer` 类 |
| `src/models/clip.py` | +24/-7 行；`lora_nsp` / `lora_sgp` 分支加 text LoRA 包装 |
| `src/trainers/lora_nsp_trainer.py` | +293/-295 行；几乎全部重写，双编码器维护 + ZS 每步重算 |
| `main_incremental.py` | +25 行；CLI + 全局 ZS 重算 |
| `main_joint.py` | +6 行；CLI（默认关 text LoRA） |
| `src/utils/main_utils.py` | +22/-14 行；批量 ZS 编码 + Transfer 索引 bugfix |
| `.gitignore` | +5/-4 行；小调整 |

**分支状态**:
```
v2-text-lora  d4242eb  v2: text LoRA-NSP + Transfer bugfix      ← HEAD
main          2711bdf  [实验] 添加 tier1 投影/初始化方法对比脚本
```

**参考**:
- 原始 commit: YongZ1999/CLIP_OOD@31cddb7 ("v2: text LoRA-NSP + Transfer bugfix", 2026-05-29)
- v2 完整 clone 保留在 `/tmp/CLIP_OOD_v2/`（本地缓存，会话外会清掉）
