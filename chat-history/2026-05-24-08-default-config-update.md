# main_incremental/joint 默认配置更新

**日期**: 2026-05-24
**会话概况**: 将 main_incremental.py 和 main_joint.py 的默认配置更新为 lora_nsp + 启用蒸馏 + 启用辅助损失

---

## 1. 默认值变更

| 参数 | 旧值 | 新值 | 文件 |
|------|:----:|:----:|------|
| `--lora_type` | `lora_vanilla` | `lora_nsp` | main_incremental.py / main_joint.py |
| `--aux_weight` | `0.0` | `1.0` | main_incremental.py |
| `--reference_dataset` | `flickr8k`（不变） | — | main_incremental.py |

`main_joint.py` 的 `--reference_dataset` 已在本次会话中从 `""` 改为 `"flickr8k"`，`--aux_weight` 保持 `0.0`（joint 模式不常用 aux head）。

## 2. 同步

- ✅ 已提交并推送到 GitHub（commit `74a0e91`）
- ❌ 服务器暂时连不上，恢复后需 `git pull`
