#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float clip(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ __forceinline__ float4 clip4(float4 v, float lo, float hi)
{
    return make_float4(clip(v.x, lo, hi), clip(v.y, lo, hi), clip(v.z, lo, hi), clip(v.w, lo, hi));
}

__global__ void clip_kernel(const float *__restrict__ in, float *__restrict__ out, size_t N, float lo, float hi)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = N / 4;

    const float4 *in4 = reinterpret_cast<const float4 *>(in);
    float4 *out4 = reinterpret_cast<float4 *>(out);

    for (size_t i = idx; i < n4; i += stride)
    {
        out4[i] = clip4(in4[i], lo, hi);
    }
    for (size_t i = n4 * 4 + idx; i < N; i += stride)
    {
        out[i] = clip(in[i], lo, hi);
    }
}

void solve(const float *in, float *out, size_t N, float lo, float hi)
{
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;

    clip_kernel<<<blocks, THREADS>>>(in, out, N, lo, hi);
}

float benchmark(const float *in, float *out, size_t N, float lo, float hi, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(in, out, N, lo, hi);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(in, out, N, lo, hi);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void clip_cpu(const float *in, float *out, size_t N, float lo, float hi)
{
    for (size_t i = 0; i < N; ++i)
    {
        out[i] = fminf(fmaxf(in[i], lo), hi);
    }
}

int main()
{
    // 创建数据
    const size_t N = 1 << 24;
    float lo = -0.5f, hi = 2.5f;
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i)
    {
        in[i] = static_cast<float>((i % 17) - 8);
    }

    // 于 CPU 上计算
    std::vector<float> ref(N);
    clip_cpu(in.data(), ref.data(), N, lo, hi);

    // 分配 GPU 内存
    float *d_in, *d_out;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);

    // 可行性验证
    solve(d_in, d_out, N, lo, hi);
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
    float average_time = benchmark(d_in, d_out, N, lo, hi, warmup_iters, bench_iters);
    double total_bytes = 2.0 * bytes;
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("计算量N = %zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", N, average_time, bandwidth_gbps);

    cudaFree(d_in);
    cudaFree(d_out);
}
