# Git 提交流程规范 + .gitignore 清理

**日期**: 2026-05-23
**会话概况**: 配置项目 Git 管理——补充 .gitignore 排除不应上传的文件，解除已误跟踪的目录，并在 AGENTS.md 中建立完整的 Git 提交策略。

---

## 1. 关键决策

- **每一次重要修改后必须提交并推送至 GitHub**，避免文件修改后无法回溯
- 这是 AI agent 的文件级记忆机制——不及时提交 = 丢失修改

## 2. 执行的改动

### .gitignore 新增排除规则

| 新增规则 | 排除内容 |
|----------|---------|
| `paper_writing/reference_papers/` | 他人论文 PDF + 截图（版权问题） |
| `paper_writing/my_original_papers/` | 个人推导笔记 PDF |
| `paper_writing/讨论/` | 历史讨论记录 |
| `paper_writing/generated-images/` | AI 生成图 |
| `paper_writing/.workbuddy/` | 本地 AI 工具配置 |
| `.workbuddy/` | 任何 .workbuddy 目录 |
| `*.pdf` | 任何编译生成的 PDF |

### git rm --cached 解除跟踪

- **`experiments/`**：190 个文件解除跟踪，保留本地
- **`optimization/`**：32 个文件解除跟踪，保留本地

### AGENTS.md 新增内容
- **第 5 节「工作守则」增加「Git 提交策略」**：提交时机、提交规范、提交信息格式
- **完整的 .gitignore 说明表格**：每个排除规则的原因

## 3. 相关文件

| 文件 | 改动 |
|------|------|
| `.gitignore` | 新增 8 条排除规则 |
| `AGENTS.md` | 新增 Git 提交策略 + .gitignore 说明 |
| `chat-history/2026-05-23-git-workflow-and-gitignore.md` | 本次记录 |

## 4. 待办事项

- [ ] 确认当前变更后可进行一次初始 commit 并 push 到 GitHub
- [ ] 后续每次重要修改后按 AGENTS.md 的 Git 策略执行 commit + push
