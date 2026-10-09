#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float4 silu4(float4 v) { return make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w)); }

__global__ void silu_kernel(const float *__restrict__ in, float *__restrict__ out, size_t N)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = N / 4;

    const float4 *in4 = reinterpret_cast<const float4 *>(in);
    float4 *out4 = reinterpret_cast<float4 *>(out);

    for (size_t i = idx; i < n4; i += stride)
    {
        out4[i] = silu4(in4[i]);
    }
    for (size_t i = n4 * 4 + idx; i < N; i += stride)
    {
        out[i] = silu(in[i]);
    }
}

void solve(const float *in, float *out, size_t N)
{
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;

    silu_kernel<<<blocks, THREADS>>>(in, out, N);
}

float benchmark(const float *in, float *out, size_t N, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(in, out, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(in, out, N);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void silu_cpu(const float *in, float *out, size_t N)
{
    for (size_t i = 0; i < N; ++i)
        out[i] = in[i] / (1.f + expf(-in[i]));
}

int main()
{
    // 创建数据
    const size_t N = 1 << 24;
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i)
    {
        in[i] = static_cast<float>((i % 13) - 6);
    }

    // 于 CPU 上计算
    std::vector<float> ref(N);
    silu_cpu(in.data(), ref.data(), N);

    // 分配 GPU 内存
    float *d_in, *d_out;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);

    // 可行性验证
    solve(d_in, d_out, N);
    cudaDeviceSynchronize();
    std::vector<float> got(N);
    cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (fabsf(ref[i] - got[i]) >
            1e-5f * (1.f + fabsf(ref[i]))) // __expf() 快速数学函数 和 expf() 标准函数 可能存在微小误差
        {
            printf("于 %zu 处不匹配, ref = %f, got = %f\n", i, ref[i], got[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "成功\n" : "错误\n");

    // 性能分析
    int warmup_iters = 10;
    int bench_iters = 100;
    float average_time = benchmark(d_in, d_out, N, warmup_iters, bench_iters);
    double total_bytes = 2.0 * bytes;
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("计算量N = %zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", N, average_time, bandwidth_gbps);

    cudaFree(d_in);
    cudaFree(d_out);
}