# LR-RGDA 符号命名规范 Prompt

## 1. 向量 (Vectors)

| 类型 | 格式 | 示例 | LaTeX |
|------|------|------|-------|
| 特征向量 | 小写粗体 | $\mathbf{x}$, $\mathbf{z}$ | `\mathbf{x}`, `\mathbf{z}` |
| 均值向量 | 希腊字母粗体 | $\boldsymbol{\mu}_c$ | `\boldsymbol{\mu}_c` |
| 权重向量 | 小写粗体 | $\mathbf{w}_c$, $\mathbf{b}_c$ | `\mathbf{w}_c`, `\mathbf{b}_c` |
| 偏移/漂移向量 | 希腊字母粗体 | $\boldsymbol{\delta}_i$ | `\boldsymbol{\delta}_i` |
| 特征表示 | 小写粗体+上下标 | $\mathbf{f}_c^{(i)}$ | `\mathbf{f}_c^{(i)}` |
| 投影向量 | 小写粗体 | $\mathbf{u}_c$ | `\mathbf{u}_c` |

## 2. 矩阵 (Matrices)

| 类型 | 格式 | 示例 | LaTeX |
|------|------|------|-------|
| 协方差矩阵 | 大写粗体Sigma | $\mathbf{\Sigma}_c$ | `\mathbf{\Sigma}_c` |
| 正则化协方差 | 大写粗体+上标 | $\mathbf{\Sigma}_c^{\text{reg}}$ | `\mathbf{\Sigma}_c^{\text{reg}}` |
| 全局协方差 | 大写粗体+下标 | $\mathbf{\Sigma}_{\text{avg}}$ | `\mathbf{\Sigma}_{\text{avg}}` |
| 单位矩阵 | 大写粗体I+下标 | $\mathbf{I}_d$ | `\mathbf{I}_d` |
| 基矩阵 | 大写粗体 | $\mathbf{B}$ | `\mathbf{B}` |
| 特征向量矩阵 | 大写粗体U | $\mathbf{U}_c$ | `\mathbf{U}_c` |
| 低秩矩阵 | 大写粗体M/Q | $\mathbf{M}_c$, $\mathbf{Q}_c$ | `\mathbf{M}_c`, `\mathbf{Q}_c` |

**矩阵运算符号**:
- 转置: $^\top$ (如 $\mathbf{U}_c^\top$)
- 逆: $^{-1}$ (如 $\mathbf{B}^{-1}$)
- 伪逆: $^{+}$ (如 $\mathbf{B}^{+}$)

## 3. 标量 (Scalars)

| 类型 | 格式 | 示例 | LaTeX |
|------|------|------|-------|
| 特征值 | 小写斜体lambda | $\lambda_i$, $\lambda_{c,i}$ | `\lambda_i`, `\lambda_{c,i}` |
| 类别索引 | 小写斜体c | $c$ | `c` |
| 特征维度 | 小写斜体d | $d$ | `d` |
| 低秩维度 | 小写斜体r | $r$ | `r` |
| 类别总数 | 大写斜体C | $C$ | `C` |
| 样本数 | 小写斜体n | $n_c$ | `n_c` |
| 温度参数 | 小写斜体tau | $\tau$ | `\tau` |
| 正则化系数 | 小写斜体alpha | $\alpha_1, \alpha_2, \alpha_3$ | `\alpha_1`, `\alpha_2`, `\alpha_3` |

## 4. 函数与运算符 (Functions & Operators)

| 类型 | 格式 | 示例 | LaTeX |
|------|------|------|-------|
| 判别函数 | 小写斜体g+上标 | $g_c^{\text{RGDA}}(\mathbf{x})$ | `g_c^{\text{RGDA}}(\mathbf{x})` |
| 马氏距离 | d^2_粗体Sigma | $d^2_{\mathbf{\Sigma}}(\mathbf{x}, \boldsymbol{\mu})$ | `d^2_{\mathbf{\Sigma}}(\mathbf{x}, \boldsymbol{\mu})` |
| 仿射函数 | 花体L | $\mathcal{L}_c(\mathbf{x})$ | `\mathcal{L}_c(\mathbf{x})` |
| 二次修正项 | 花体Q | $\mathcal{Q}_c(\mathbf{x})$ | `\mathcal{Q}_c(\mathbf{x})` |
| Softmax | 标准函数 | $p_\varphi(\cdot; \mathcal{S})$ | `p_\varphi(\cdot; \mathcal{S})` |

## 5. 集合 (Sets)

| 类型 | 格式 | 示例 | LaTeX |
|------|------|------|-------|
| 类别集合 | 花体C | $\mathcal{C}_t$ | `\mathcal{C}_t` |
| 历史统计集合 | 花体H | $\mathcal{H}_t$ | `\mathcal{H}_t` |
| 标签子集 | 花体S | $\mathcal{S}$ | `\mathcal{S}` |

## 6. 希腊字母使用规范

| 符号 | 用途 | 示例 |
|------|------|------|
| $\mu$ (mu) | 均值向量 | $\boldsymbol{\mu}_c$ |
| $\Sigma$ (Sigma) | 协方差矩阵 | $\mathbf{\Sigma}_c$ |
| $\alpha$ (alpha) | 正则化系数 | $\alpha_1, \alpha_2, \alpha_3$ |
| $\lambda$ (lambda) | 特征值 | $\lambda_i$ |
| $\delta$ (delta) | 漂移/偏移 | $\boldsymbol{\delta}_i$ |
| $\tau$ (tau) | 温度参数 | $\tau$ |
| $\pi$ (pi) | 先验概率 | $\pi_c$ |

## 7. 常用公式模板

### 均值与协方差估计
```latex
\boldsymbol{\mu}_c = \frac{1}{n_c}\sum_{i=1}^{n_c}\mathbf{f}_c^{(i)}

\mathbf{\Sigma}_c = \frac{1}{n_c}\sum_{i=1}^{n_c}\mathbf{f}_c^{(i)}(\mathbf{f}_c^{(i)})^\top - \boldsymbol{\mu}_c\boldsymbol{\mu}_c^\top
```

### 正则化协方差
```latex
\mathbf{\Sigma}_c^{\text{reg}} = \alpha_1\mathbf{\Sigma}_c + \alpha_2\mathbf{\Sigma}_{\text{avg}} + \alpha_3\mathbf{I}_d
```

### RGDA判别函数
```latex
g_c^{\text{RGDA}}(\mathbf{x}) = -\frac{1}{2}d^2_{\mathbf{\Sigma}_c^{\text{reg}}}(\mathbf{x}, \boldsymbol{\mu}_c) - \frac{1}{2}\log\det(\mathbf{\Sigma}_c^{\text{reg}})

\text{其中 } d^2_{\mathbf{\Sigma}}(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^\top\mathbf{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})
```

### LDA判别函数
```latex
g_c^{\text{LDA}}(\mathbf{x}) = \mathbf{w}_c^\top\mathbf{x} + b_c

\mathbf{w}_c = \mathbf{\Sigma}_{\text{avg}}^{-1}\boldsymbol{\mu}_c

b_c = -\frac{1}{2}\boldsymbol{\mu}_c^\top\mathbf{\Sigma}_{\text{avg}}^{-1}\boldsymbol{\mu}_c
```

### Woodbury矩阵恒等式
```latex
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}^\top)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}^\top\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^\top\mathbf{A}^{-1}
```

## 8. 注意事项

1. **粗体使用**: 所有向量和矩阵必须使用 `\mathbf{}` 或 `\boldsymbol{}` 加粗
2. **转置符号**: 统一使用 `^\top` 而非 `^T`
3. **上下标位置**: 类别索引用下标（如 $_c$），方法名称用上标（如 $^{\text{reg}}$）
4. **花体字母**: 集合和特定函数使用 `\mathcal{}`
5. **数学环境**: 公式块使用 `equation` 环境，对齐使用 `align` 环境

## 9. 一致性检查清单

- [ ] 向量是否全部加粗？
- [ ] 矩阵是否全部加粗？
- [ ] 转置是否使用 `^\top`？
- [ ] 希腊字母向量是否使用 `\boldsymbol`？
- [ ] 集合是否使用 `\mathcal`？
- [ ] 标量是否保持斜体（不加粗）？
