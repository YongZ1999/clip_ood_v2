# chat-history 文件命名规范统一

**日期**: 2026-05-23
**会话概况**: 统一 chat-history 文件名格式，同一天的日志增加两位序号，保持目录整洁有序。

---

## 1. 关键决策

- **文件命名格式**：`{YYYY-MM-DD}-{seq}-{topic}.md`，其中 `seq` 为两位序号（01, 02, 03...）
- **排序规则**：同一天的多条记录按时间先后递增序号
- AGENTS.md 中的示例列表和命名规则同步更新

## 2. 改动内容

### chat-history 目录重命名（5 个文件）

| 旧文件名 | 新文件名 |
|----------|----------|
| `2026-05-23-project-overview.md` | `2026-05-23-01-project-overview.md` |
| `2026-05-23-agents-md-creation.md` | `2026-05-23-02-agents-md-creation.md` |
| `2026-05-23-paper-writing-and-lada-baseline.md` | `2026-05-23-03-paper-writing-and-lada-baseline.md` |
| `2026-05-23-git-workflow-and-gitignore.md` | `2026-05-23-04-git-workflow-and-gitignore.md` |
| `2026-05-23-server-workflow-and-repo-rename.md` | `2026-05-23-05-server-workflow-and-repo-rename.md` |

### AGENTS.md 更新
- 文件命名规则模板增加 `{seq}` 字段
- chat-history 示例列表更新为新文件名

## 3. 相关文件

| 文件 | 改动 |
|------|------|
| `chat-history/` | 5 个文件重命名 |
| `AGENTS.md` | 命名规则 + 示例列表更新 |
