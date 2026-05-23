# 在 project_clip_continual_learning 中创建 AGENTS.md

**日期**: 2026-05-23
**会话概况**: 用户要求在 `project_clip_continual_learning` 目录下创建一份专门给 AI agent 阅读的指南文档，并建立保存对话重点到 `chat-history/` 的机制。

---

## 1. 关键决策

- **AGENTS.md 位置**: 放在 `project_clip_continual_learning/` 根目录
- **保存对话记录的职责**: 明确要求 AI agent 在每次有意义沟通后，将对话重点写入 `chat-history/{YYYY-MM-DD}-{topic}.md`
- **内容范围**: AGENTS.md 包含项目快速概览（省去 AI 每次重新探索）、代码结构速查、关键超参数、工作守则、重要约定

## 2. 产出文件

| 文件 | 说明 |
|------|------|
| `AGENTS.md` | AI agent 专属指南（含项目概览、代码速查、工作守则、以及**保存 chat-history 的核心职责**） |
| `chat-history/2026-05-23-agents-md-creation.md` | 本记录 |

## 3. 待办事项

- [ ] 后续 AI agent 进入项目时，应先读 `AGENTS.md` + 最近 `chat-history/` 记录
- [ ] AGENTS.md 中提到的架构/参数/结构若有变动，需同步更新该文件
