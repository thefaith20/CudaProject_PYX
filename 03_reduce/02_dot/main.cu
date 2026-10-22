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
// dot 算子内核
// ============================================================

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__global__ void dot_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ out, size_t N)
{
    float part = 0.f;
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = N / 4;

    const float4 *a4 = reinterpret_cast<const float4 *>(A);
    const float4 *b4 = reinterpret_cast<const float4 *>(B);

    for (size_t i = idx; i < n4; i += stride)
    {
        part += dot4(a4[i], b4[i]);
    }
    for (size_t i = n4 * 4 + idx; i < N; i += stride)
    {
        part += A[i] * B[i];
    }

    part = blockReduceSum<BLOCK>(part);
    if (threadIdx.x == 0)
    {
        atomicAdd(out, part);
    }
}

// ============================================================
// solve 包装
// ============================================================

void solve(const float *d_A, const float *d_B, float *d_out, size_t N)
{
    cudaMemset(d_out, 0, sizeof(float)); // 清零
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 2 : 128;
    int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;
    dot_kernel<<<blocks, BLOCK>>>(d_A, d_B, d_out, N);
}

double dot_cpu(const float *A, const float *B, size_t N)
{
    double result = 0.f;
    for (size_t i = 0; i < N; ++i)
    {
        result += A[i] * B[i];
    }
    return result;
}

// ============================================================
// 测试框架
// ============================================================

void run_and_bench(const float *d_A, const float *d_B, float *d_out, size_t N, double cpu_dot, int warmup, int bench)
{
    solve(d_A, d_B, d_out, N);
    cudaDeviceSynchronize();
    float gpu_dot = 0.f;
    cudaMemcpy(&gpu_dot, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    double rel_err = fabs((double)gpu_dot - cpu_dot) / (1.0 + fabs(cpu_dot));
    if (rel_err > 1e-4)
    {
        printf("计算错误, cpu_dot = %.6f, gpu_dot = %.6f\n", cpu_dot, gpu_dot);
        return;
    }

    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < warmup; ++i)
    {
        solve(d_A, d_B, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench; ++i)
    {
        solve(d_A, d_B, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float ms_average = ms_total / bench;
    double total_bytes = static_cast<double>(N * 2 * sizeof(float));
    double bandwidth = total_bytes / (ms_average * 1e-3) / 1e9;

    printf("N = %zu, dot = %.6f, 耗时 = %.4f ms, 有效带宽 = %.1f GB/s\n", N, gpu_dot, ms_average, bandwidth);
}

// ============================================================
// main 函数
// ============================================================

int main()
{
    const size_t N = 1 << 24;
    size_t bytes = N * sizeof(float);
    std::vector<float> A(N), B(N);
    for (size_t i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>((i % 17) - 8);
        B[i] = static_cast<float>((i % 13) - 6);
    }
    double cpu_dot = dot_cpu(A.data(), B.data(), N);

    float *d_A, *d_B, *d_out;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    const int warmup = 10;
    const int bench = 100;
    run_and_bench(d_A, d_B, d_out, N, cpu_dot, warmup, bench);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}