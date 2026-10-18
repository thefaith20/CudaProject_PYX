# 02 permute 算子性能测试

## 测试环境
- **GPU**: NVIDIA GeForce RTX 3080 Ti
- **理论峰值带宽**: 912 GB/s
- **测试方法**: `cudaEvent` 预热 10 次，循环 100 次取平均
- **数据类型**: FP32 (单精度浮点)

## 性能总览表

| 算子 (Operator) | 类型  | 访存模式 (Memory Pattern) | 核心优化策略 | 有效带宽 (GB/s) | 达成率 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Interleave** | 访存型 | 连续读 & 连续写 | float4 向量化 | **806.2** | 88.4% | 纯合并访问，触及硬件极限 |
| **Transpose (naive)** | 访存型 | 碎片化写 (Scattered) | 纯多维网格映射 | 251.4 | 27.6% | 写地址严重碎片化 |
| **Transpose (shared)** | 访存型 | 跨步转置 (Strided) | Shared Mem + Padding |  705.0 | 77.3% | 消除写碎片，TILE 和 Occupancy 平衡 |
| **Transpose (shared + float4)** | 访存型 | 跨步转置 (Strided) | Shared Mem + Padding + 向量化 | **788.3** | 86.4% | 原基础上向量化，完美合并读写 |
| **NCHW→NHWC (Native)**| 访存型| 碎片化写 (Scattered) | 纯多维网格映射 | 103.7 | 11.3% | 写地址严重碎片化，触发显存灾难 |
| **NCHW→NHWC (Shared)**| 访存型 | 连续读 & 连续写 | Shared Mem 矩阵转置法 |  **791.8** | 86.8% | 彻底消除写碎片，性能暴涨 |

## 性能数据分析 (Insights)

1. **维度重排的代价 (Layout Transform Cost)**：
   在 `NCHW -> NHWC` 的朴素实现中，由于相邻线程在输出时的地址跨度过大，导致全局内存写操作完全无法合并 。Shared Memory 充当了完美的“中转站”，通过 `[C, HW] -> [HW, C]` 的矩阵转置，实现了读写双完美合并，性能从 **103.7 GB/s 飙升至 791.8 GB/s**。

2. **Occupancy 与 Tile 大小的权衡**：
   在 Matrix Transpose 中，如果将 `TILE` 设置为 32（Block 包含 1024 个线程），虽然理论上合并访存完美，但由于单个 Block 过大，导致 SM（流多处理器）只能同时驻留极少的 Block。一旦遇到内存延迟，SM 便无任务可调度。
   当把 `TILE` 缩小到 **16**（Block 包含 256 个线程）时，SM 可以同时运行更多的 Block（高 Occupancy）。大量并行的 Warp 成功掩盖了内存延迟，使得带宽从 576 GB/s 提升至 **705 GB/s**。

3. **Bank Conflict 消除**：
   所有使用 Shared Memory 的 Transpose 算子，均采用了 `smem[TILE][TILE + 1]` 的 Padding 技巧。通过多消耗极小的显存（几个字节），彻底打散了跨步访问时带来的 Bank 冲突，保证了片上缓存的极限吞吐。