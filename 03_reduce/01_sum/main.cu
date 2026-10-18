#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int BLOCK = 256;

// ============================================================
// Reduce 核心工具
// ============================================================

// warp 内归约：线程内寄存器开销极低
__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (size_t offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(0xffffffff, v, offset); // 线程 i 把线程 (i ^ offset) 的 v 拿过来，加到自己身上
    }
    return v;
}

// block 内归约：warp 内先 shuffle 归约，warp 间用极小的共享内存
template <size_t BLOCK_SIZE> __device__ __forceinline__ float blockReduceSum(float v)
{
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    size_t lane = threadIdx.x & 31; // warp 内 thread 的编号
    size_t warp = threadIdx.x >> 5; // block 内 warp 的编号

    v = warpReduceSum(v);

    // 只获取每个 warp 内第一个 thread 的数据
    if (lane == 0)
    {
        warp_sums[warp] = v;
    }
    __syncthreads();
    // 只在第一个 warp 内的 threads 计算
    if (warp == 0)
    {
        v = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.f;
        v = warpReduceSum(v);
    }
    return v;
}

// ============================================================
// 算子内核
// ============================================================

__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }

__global__ void reduce_sum_kernel(const float *__restrict__ in, float *__restrict__ out, size_t N)
{
    float part = 0.f;

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t n4 = N / 4;

    const float4 *in4 = reinterpret_cast<const float4 *>(in);

    // float4 向量化主循环
    for (size_t i = idx; i < n4; i += stride)
    {
        part += sum4(in4[i]);
    }
    // 标量尾循环
    for (size_t i = n4 * 4 + idx; i < N; i += stride)
    {
        part += in[i];
    }

    part = blockReduceSum<BLOCK>(part); // 每个 block 计算的总和
    // 每个 block 的 lane 0 写回全局（原子加）
    if (threadIdx.x == 0)
    {
        atomicAdd(out, part);
    }
}

// ============================================================
// solve 包装
// ============================================================

void solve(const float *d_in, float *d_out, size_t N)
{
    cudaMemset(d_out, 0, sizeof(float)); // d_清零

    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 2 : 128; // 硬件自适应
    const int need = CEIL(N, BLOCK);
    if (need < blocks) // 防止浪费
        blocks = need;
    if (blocks == 0) // 防止崩溃
        blocks = 1;

    reduce_sum_kernel<<<blocks, BLOCK>>>(d_in, d_out, N);
}

// ============================================================
// 测试框架
// ============================================================

// CPU 参考：用 double 减少累积误差
double reduce_sum_cpu(const float *in, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += static_cast<double>(in[i]);
    return sum;
}

// 测试函数
void run_and_bench(const float *d_in, float *d_out, size_t N, double cpu_sum, int warmup, int bench)
{
    // 正确性验证
    solve(d_in, d_out, N);
    cudaDeviceSynchronize();
    float gpu_sum = 0.f;
    cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    double rel_err = fabs((double)gpu_sum - cpu_sum) / (1.0 + fabs(cpu_sum));
    if (rel_err > 1e-4)
    {
        printf("误差过大, cpu_sum = %.6f, gpu_sum = %.6f\n", cpu_sum, gpu_sum);
        return;
    }

    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < warmup; ++i)
    {
        solve(d_in, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench; ++i)
    {
        solve(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float ms_average = ms_total / bench;
    double read_bytes = static_cast<double>(N * sizeof(float));
    double read_bandwidth = read_bytes / (ms_average * 1e-3) / 1e9;

    printf("N = %zu, sum = %.6f, 耗时 = %.4f ms, 读带宽 = %.1f GB/s\n", N, gpu_sum, ms_average, read_bandwidth);
}

// ============================================================
// main
// ============================================================

int main()
{
    const size_t N = 1 << 24;
    size_t bytes = N * sizeof(float);
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i)
        in[i] = static_cast<float>((i % 1000) * 0.001 - 0.5);

    double cpu_sum = reduce_sum_cpu(in.data(), N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);

    const int warmup = 10;
    const int bench = 100;
    run_and_bench(d_in, d_out, N, cpu_sum, warmup, bench);

    cudaFree(d_in);
    cudaFree(d_out);
}