# CUDA Operator Implementation & Optimization 
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

个人从零开始学习并手写 CUDA 底层算子的完整过程，后续会一直持续更新，欢迎各位一起学习。

每个算子不仅包含**基于 CUDA 的原生实现**，还包含**性能基准测试（Benchmark）**、**优化历程对比**、**性能差异分析**等。


##  目录结构 (Project Structure)

目前涵盖以下算子分类（持续更新中）：

- [x] [01_elementwise](./01_elementwise) - 逐元素操作 (Add, ReLU, Sigmoid, SiLU, etc.)
- [x] [02_permute](./02_permute) - 维度重排 (Reverse, Transpose, Interleave, etc.)
- [x] [03_reduce](./03_reduce) - 归约操作 (Reduction, Softmax, Dot, etc.)
- [ ] [04_gemm](./04_gemm) - 矩阵乘法 
- [ ] [05_stencil](./05_stencil) - 模板计算 
- [ ] [06_attention](./06_attention) - 注意力机制 


##  致谢 (Acknowledgements)

本仓库的学习路线、部分基础代码框架及 Benchmark 测试方法，受到以下优秀开源项目和老师的启发，站在巨人的肩膀上学习：
- [ZiyuanDu](https://github.com/ZiyuanDu/Awesome_AI_Infra_Learning) - 优秀的学习路线和测试框架。
- [ZiyuanDu](https://dlog.com.cn/) - 杜子源老师的优质 AI Infra 系列教程。
- [xlite-dev](https://github.com/xlite-dev/LeetCUDA) - 丰富的各种类 CUDA 算子。
- [LeetGPU](https://leetgpu.com/challenges) - 各个难度的 CUDA 算子挑战。
- [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) - 经典的 CUDA 优化教程。
- [OneFlow](https://github.com/Oneflow-Inc/oneflow) - 优秀的 elementwise 框架设计。


