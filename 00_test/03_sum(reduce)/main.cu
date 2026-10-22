#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int BLOCK = 1024;
constexpr int GRID_BLOCKS = 1024; // v6/v7 用的固定 block 数

// ============================================================
// v0：交错寻址 + 取模，问题：warp divergence + 取模运算慢
// ============================================================
__global__ void reduce_v0(const float *in, float *out, int N)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

// ============================================================
// v1：乘法代替取模，但仍有 bank conflict
// ============================================================
__global__ void reduce_v1(const float *in, float *out, int N)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
            sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

// ============================================================
// v2：顺序寻址（关键改进，消除 bank conflict）
// ============================================================
__global__ void reduce_v2(const float *in, float *out, int N)
{
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = BLOCK >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// v3：每线程加载两个元素（block 数减半）
// ============================================================
__global__ void reduce_v3(const float *in, float *out, int N)
{
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    float val1 = (idx < N) ? in[idx] : 0.0f;
    float val2 = (idx + blockDim.x < N) ? in[idx + blockDim.x] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();
    for (int s = BLOCK >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// v4：warp 内免同步（volatile + 手动展开）
// ============================================================
__device__ void warpReduce_v4(volatile float *smem, int tid)
{
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduce_v4(const float *in, float *out, int N)
{
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    float val1 = (idx < N) ? in[idx] : 0.0f;
    float val2 = (idx + blockDim.x < N) ? in[idx + blockDim.x] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();
    for (int s = BLOCK >> 1; s > 32; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32)
        warpReduce_v4(smem, tid);
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// v5：模板化 warpReduce，编译期消除分支
// ============================================================
template <unsigned int blockSize> __device__ void warpReduce_v5(volatile float *smem, int tid)
{
    if (blockSize >= 64)
        smem[tid] += smem[tid + 32];
    if (blockSize >= 32)
        smem[tid] += smem[tid + 16];
    if (blockSize >= 16)
        smem[tid] += smem[tid + 8];
    if (blockSize >= 8)
        smem[tid] += smem[tid + 4];
    if (blockSize >= 4)
        smem[tid] += smem[tid + 2];
    if (blockSize >= 2)
        smem[tid] += smem[tid + 1];
}

template <unsigned int blockSize> __global__ void reduce_v5_impl(const float *in, float *out, int N)
{
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + tid;
    float val1 = (idx < N) ? in[idx] : 0.0f;
    float val2 = (idx + blockSize < N) ? in[idx + blockSize] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
            smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            smem[tid] += smem[tid + 64];
        __syncthreads();
    }
    if (tid < 32)
        warpReduce_v5<blockSize>(smem, tid);
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// v6：grid-stride 循环，适应任意 N
// ============================================================
template <unsigned int blockSize> __global__ void reduce_v6_impl(const float *in, float *out, int N)
{
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + tid;
    int gridSize = blockSize * 2 * gridDim.x;
    float sum = 0.0f;
    while (idx < N)
    {
        sum += in[idx];
        if (idx + blockSize < N)
            sum += in[idx + blockSize];
        idx += gridSize;
    }
    smem[tid] = sum;
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
            smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            smem[tid] += smem[tid + 64];
        __syncthreads();
    }
    if (tid < 32)
        warpReduce_v5<blockSize>(smem, tid);
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// v7：float4 向量化 + grid-stride
// ============================================================
template <unsigned int blockSize>
__global__ void reduce_v7_impl(const float *__restrict__ in, float *__restrict__ out, int N)
{
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    const float4 *in4 = reinterpret_cast<const float4 *>(in);
    int n_vec = N / 4;
    int idx = blockIdx.x * blockSize + tid;
    int grid_stride = gridDim.x * blockSize;
    float sum = 0.0f;
    for (int i = idx; i < n_vec; i += grid_stride)
    {
        float4 val = in4[i];
        sum += val.x + val.y + val.z + val.w;
    }
    // 尾部处理：只让 block 0 的 thread 0 干
    if (blockIdx.x == 0 && tid == 0)
    {
        for (int i = n_vec * 4; i < N; ++i)
            sum += in[i];
    }
    smem[tid] = sum;
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
            smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            smem[tid] += smem[tid + 64];
        __syncthreads();
    }
    if (tid < 32)
        warpReduce_v5<blockSize>(smem, tid);
    if (tid == 0)
        atomicAdd(out, smem[0]);
}

// ============================================================
// CPU 参考：用 double 减少累积误差
// ============================================================
float reduce_cpu(const float *in, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += (double)in[i];
    return (float)sum;
}

// ============================================================
// 测试模板函数
// ============================================================
template <typename LaunchFn>
void run_and_bench(const char *name, LaunchFn launch, float *d_out, float cpu_sum, int warmup, int bench)
{
    cudaMemset(d_out, 0, sizeof(float));
    launch();
    cudaDeviceSynchronize();
    float got = 0.0f;
    cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float rel_err = fabsf(got - cpu_sum) / (1.0f + fabsf(cpu_sum));
    bool ok = (rel_err < 1e-3f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < warmup; ++i)
    {
        cudaMemset(d_out, 0, sizeof(float));
        launch();
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench; ++i)
    {
        cudaMemset(d_out, 0, sizeof(float));
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("%-4s sum = %10.4f  rel_err = %.2e  %s  耗时 = %8.4f ms\n", name, got, rel_err, ok ? "OK  " : "FAIL",
           ms_total / bench);
}

int main()
{
    const int N = 1 << 24; // 16M 个 float = 64 MB
    const size_t bytes = (size_t)N * sizeof(float);

    // 1. 准备输入数据
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)((i % 1000) * 0.001 - 0.5);

    // 2. CPU 参考值
    float cpu_sum = reduce_cpu(h_in.data(), N);
    printf("CPU sum = %.6f\n\n", cpu_sum);

    // 3. 分配显存
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    const int warmup = 10;
    const int bench = 100;

    int blocks_n = CEIL(N, BLOCK);
    int blocks_2x = CEIL(N, BLOCK * 2);

    // 4. 性能测试
    run_and_bench("v0", [&] { reduce_v0<<<blocks_n, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench("v1", [&] { reduce_v1<<<blocks_n, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench("v2", [&] { reduce_v2<<<blocks_n, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench("v3", [&] { reduce_v3<<<blocks_2x, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench("v4", [&] { reduce_v4<<<blocks_2x, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench(
        "v5", [&] { reduce_v5_impl<BLOCK><<<blocks_2x, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench(
        "v6", [&] { reduce_v6_impl<BLOCK><<<GRID_BLOCKS, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
    run_and_bench(
        "v7", [&] { reduce_v7_impl<BLOCK><<<GRID_BLOCKS, BLOCK>>>(d_in, d_out, N); }, d_out, cpu_sum, warmup, bench);
}