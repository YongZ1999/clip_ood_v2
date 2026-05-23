# 项目进展回顾与问题修复（问题1-3）

**日期**: 2026-05-24
**会话概况**: 用户要求重新阅读项目全部内容（含 12 条 chat-history），了解项目当前进展，并修复遗留问题 1-3。

---

## 1. 关键决策 / 修复

### 问题1: `main_incremental.py` lora_type choices 未同步
- **问题**: 旧代码 `choices=["lora_sgp", "lora_nsp"]`, `default="lora_nsp"`
- **修复**: 改为 `choices=["lora_vanilla", "lora_sgp", "lora_nsp"]`, `default="lora_vanilla"`
- **对齐**: 与 `main_joint.py` 保持一致

### 问题2: 未提交的更改
- 涉及 4 个文件修改 + 4 个新 chat-history 文件
- 统一提交于 commit `1fe22b0`

### 问题3: 未推送的 commit
- 推送到 `origin/main`（中间遇到 HTTP2 framing error，临时切为 HTTP/1.1 后成功，已恢复配置）

## 2. 合并的修改内容

| 文件 | 改动 |
|------|------|
| `main_incremental.py` | lora_type choices/default 对齐 main_joint.py |
| `main_joint.py` | 评估流程重构（run_full_evaluation）+ 中间评估回调 + OOD 修复 + 更多 |
| `src/trainers/lora_nsp_trainer.py` | EMA 损失/准确率记录 + eval_interval/eval_callback |
| `src/utils/reference_loader.py` | 修复蒸馏跳过条件 |
| `chat-history/` | 新增第09-12条记录 |

## 3. 当前项目状态

**总体完成度**: ~85%（核心算法全部就绪）

| 类别 | 状态 |
|------|------|
| 核心算法（LoRA-NSP, LR-RGDA, 自适应路由） | ✅ ~100% |
| Phase 1 实验（推理端不微调） | ✅ 完成 |
| Phase 3-4 实验（训练端微调） | ⚠️ ~60%（框架就位，部分未跑完） |
| 超参数优化 | ✅ ~100% |
| 消融实验框架 | ⚠️ ~50% |
| 代码清理 | ⚠️ ~60% |
| 单元测试 | ⚠️ ~20% |
| 论文写作（实验数据占位符、方法图） | ⚠️ 待完成 |

## 4. 剩余遗留问题

- [ ] 仓库重命名（`clip_ood` → `project_clip_continual_learning`）
- [ ] 远程服务器实验结果查看
- [ ] 消融实验框架完善
- [ ] 代码清理（旧版 `models/`, `classifier/`）
- [ ] 单元测试
- [ ] 论文实验数据替换 + 方法图制作

## 5. 相关文件

- `AGENTS.md`: AI agent 指南（含项目概览、工作守则、Git 策略）
- `PROJECT_DOCUMENTATION.md`: 完整项目文档和 API 参考
- `PRESENTATION.md`: 研究汇报
