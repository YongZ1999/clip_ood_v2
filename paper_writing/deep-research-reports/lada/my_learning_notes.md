# 我的学习笔记

---

## Paper 1: LADA: Scalable Label-Specific CLIP Adapter for Continual Learning (Done)

## Paper 2: ZSCL: Preventing Zero-Shot Transfer Degradation in Continual Learning (ICCV 2023)
- **核心贡献**：首次系统定义了“转移退化”（Forward Forgetting）。提出使用**参考数据集**进行特征空间蒸馏，并结合参数空间的**权重集成（WE）**。
- **关键洞察**：维持预训练拓扑结构比单纯记忆旧样本更重要。
- **与本项目连接**：提供了基础的 MTIL 评估框架和 Transfer 指标。

## Paper 3: MoE-Adapters: Boosting CL via Mixture-of-Experts Adapters (CVPR 2024)
- **核心贡献**：引入 MoE 结构，将 LoRA 作为专家。提出 **DDAS（自动选择器）** 解决推理时对 Task-ID 的依赖。
- **关键洞察**：单一适配器容量有限，MoE 能提供更大的可塑性空间。
- **与本项目连接**：DDAS 是处理无身份推理的重要参考。

## Paper 4: RAIL: Advancing Cross-domain Discriminability (NeurIPS 2024)
- **核心贡献**：提出 **X-TAIL** 设定（跨域无身份）。利用**岭回归解析解**实现绝对记忆，通过**高维非线性投影**解决域间干扰。
- **关键洞察**：解析解能彻底消除梯度漂移；高维空间能解耦语义重叠。
- **与本项目连接**：X-TAIL 是本项目最核心的测试场景；高维投影思路启发了 NSP 等优化。

---

## 整体框架更新：CLIP 持续学习的技术演进

1. **第一阶段：特征空间正则化 (2023)**
   - 代表：ZSCL
   - 目标：保住预训练知识。
   - 手段：蒸馏、参考集、权重平均。

2. **第二阶段：架构扩展与任务识别 (2024 初)**
   - 代表：MoE-Adapters
   - 目标：增加模型容量，自动推断任务身份。
   - 手段：MoE (LoRA)、自编码器 (DDAS)。

3. **第三阶段：解析解与跨域解耦 (2024 末)**
   - 代表：RAIL
   - 目标：绝对记忆，解决跨域相关性导致的分类错误。
   - 手段：岭回归递推解、RHL/Kernel 投影、OOD 融合。

4. **第四阶段：标签特定与轻量化 (2025)**
   - 代表：LADA
   - 目标：极致的 Scalability，解决 RAIL 随类别增长的存储问题。
   - 手段：标签特定记忆单元（Label-specific memory）、GMM 分布保持。
