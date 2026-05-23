# AGENTS.md 新增开发与运行环境章节 + GitHub 仓库重命名

**日期**: 2026-05-23
**会话概况**: 明确项目开发工作流——本地只保存代码，代码实际在远程 GPU 服务器上运行。同时讨论 GitHub 仓库从 clip_ood 重命名为 project_clip_continual_learning。

---

## 1. 关键决策

- **开发架构确认**：本地 Mac 仅保存和编辑代码，实际运行在服务器 `raoxuan@10.20.34.30`（路径 `/home/raoxuan/projects/clip_ood`）
- **AI agent 工作流**：每次需要执行代码时，必须先 `git push`，再 SSH 到服务器 `git pull` 后执行
- **GitHub 仓库重命名**：从 `clip_ood` → `project_clip_continual_learning`（需在 GitHub 网页上操作，之后更新本地 remote URL）

## 2. AGENTS.md 变更

- 新增 **第 3 节「开发与运行环境」**：
  - 服务器信息表（地址、远程路径、数据路径）
  - 标准工作流图示（本地编辑→git push→服务器 pull→运行）
  - AI agent 执行代码/实验的具体命令模板
  - 注意事项（不要本地运行、提交前确保可运行、服务器已装好依赖等）
  - GitHub 仓库重命名说明
- 修复了之前编辑造成的章节编号错乱问题

## 3. 待办事项

- [ ] 用户需在 GitHub 上将仓库 `clip_ood` 重命名为 `project_clip_continual_learning`
- [ ] 重命名后运行 `git remote set-url origin https://github.com/raoxuan98-hash/project_clip_continual_learning.git`
- [ ] 可检查服务器上 `.gitignore` 是否与本地一致
