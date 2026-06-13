# LADA GMM 特征空间更正

## 官方实现结论

LADA 的 DPT GMM 在 CLIP 图像编码器输出的未归一化 embedding 空间拟合和采样。
采样的旧类特征与当前图像特征拼接后，再统一进行 L2 归一化并输入文本分类器和
LADA 分类器。LADA 自身的中心初始化和分类发生在单位球面空间。

## 本项目检查结果

- `main_joint.py` 虽然保留了名为 `all_raw_features` 的张量，但该张量来自默认执行
  L2 归一化的 `extract_features()`，因此实际上仍是球面特征。进一步追踪调用链后
  已在后续修改中修复该问题。
- `main_incremental_lada.py` 和 `LADATrainer` 此前默认在拟合 DPT GMM 前归一化特征，
  且将原始空间回放样本与已归一化当前样本直接拼接。这与官方实现不一致。

## 已完成修正

- DPT GMM 默认使用未归一化 CLIP embedding；官方模式强制关闭拟合前归一化。
- 训练时先拼接原始空间的 DPT 回放和当前图像特征，再统一调用 `F.normalize`。
- 官方 DPT 消融脚本将 `image_prototypes_weight_coef` 从 `1.0` 恢复为 `64.0`。

## 本地验证

- Python 静态编译检查通过。
- `git diff --check` 通过。
- 合成特征测试中，GMM 均值范数保留为约 `2.96` 和 `4.99`；分类前归一化后的
  回放特征范数均为 `1.0`。
