# Joint 分类器回放实验修正与执行准备

## 目标

在 `main_joint.py` 中使用完全相同的真实或 GMM 伪特征，公平比较 LADA 与
LR-RGDA。核心条件包括真实特征、官方原始空间 GMM、球面空间 GMM 错误对照和
原始空间 GMM 均值回放。

## 新发现的历史实现问题

`src/utils/feature_extractor.py::extract_features()` 原本固定执行 L2 归一化。
`main_joint.py` 将其返回值复制为 `all_raw_features`，但该张量实际上已经位于单位
球面。因此，历史记录中标为 raw-space GMM 的结果实际上是 sphere-space GMM。

另外，旧 `LR-RGDA-GMM` 仅使用 GMM 样本微调 affine 参数，LR-RGDA 的解析中心和
协方差仍由真实特征构建。该配置在模拟历史真实特征不可访问时存在信息泄漏，且与
完全由伪样本构建的 LADA 不公平。

因此，2026-05-31 记录的 GMM 数字不能作为官方 raw-space GMM 的结论，需要重跑。

## 已完成修改

- `extract_features()` 新增向后兼容的 `normalize` 参数。
- `main_joint.py` 显式提取 raw CLIP embedding，并从同一 raw 张量派生单位球面特征。
- 新增 `--gmm_fit_space raw|sphere`。
- 新增 `--gmm_sample_mode sample|mean`。
- GMM 条件下，LR-RGDA 和 LADA 均从同一批归一化伪样本构建并微调。
- 真实条件下，两者均从同一批真实单位球面特征构建并微调。
- 结果 JSON 增加实验名称，并支持指定输出目录。

## 待运行配置

脚本：`scripts/run_joint_classifier_replay.sh`

默认运行 3 个随机种子：

1. `real`
2. `gmm_raw`
3. `gmm_sphere`
4. `gmm_raw_mean`

每次运行同时输出 CLIP-ZS、LR-RGDA、LADA、LR-RGDA+ZS 和 LADA+ZS。汇总器将生成
`summary.csv` 与 `summary.md`，结果格式为均值加标准差。

## 本地验证

- Python 编译检查通过。
- Shell 语法检查通过。
- `git diff --check` 通过。
- 假模型测试确认 raw 特征范数保持为 `3.0/6.0`，归一化路径输出范数均为 `1.0`。
- 本机缺少 `transformers`，因此完整 `main_joint.py` 入口需要在服务器环境运行。
