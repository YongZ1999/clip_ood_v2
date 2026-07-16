# 真实官方 LADA 跨模态检索结果导入

**日期**: 2026-07-16  
**来源文件**: 用户从服务器导出的 `retrieval42.json`、`retrieval43.json`、`retrieval44.json`。

## 完整性核验

三个文件均通过以下检查：

- `protocol == official_lada_in_process_text_tuner`；
- 总行数 40 = `10 tasks × 2 datasets × 2 methods (lada/frozen_clip)`；
- 每个 seed、每个数据集均有 10 行 `method=lada`；
- `lada_summary[dataset].num_tasks == 10`。

Frozen CLIP 行在 3 个 seed 中完全一致，且与已有第 7 节基线完全一致：MSCOCO I2T/T2I R@1=52.32/33.3267，Flickr30K=81.10/60.78。这验证了官方 LADA 评测与项目既有检索协议使用了相同的冻结参考数值。

## 三 seed 汇总（mean ± sample std, ddof=1）

| Dataset | Avg I2T R@1 | Avg T2I R@1 | Last I2T R@1 | Last T2I R@1 |
|---|:---:|:---:|:---:|:---:|
| MSCOCO 5K | 51.14 ± 0.20 | 30.17 ± 0.29 | 53.81 ± 0.19 | 24.28 ± 0.92 |
| Flickr30K | 79.53 ± 0.18 | 58.19 ± 0.49 | 81.43 ± 0.12 | 51.91 ± 1.23 |

## 解释

官方 LADA 的视觉编码器冻结，但每个任务训练当前 Text AdaptFormer。真实结果表明：最终 I2T 略高于 Frozen CLIP（COCO +1.49，Flickr +0.33），而 T2I 显著低于 Frozen CLIP（COCO -9.05，Flickr -8.87）。因此此前 “Frozen CLIP = LADA retrieval” 既不成立，也会掩盖 LADA 的方向性检索退化。

`docs/paper_experiment_results.md` 第 7 节已替换为上述真实 LADA 行；原 Frozen CLIP 保留为实际测得的独立对照。
