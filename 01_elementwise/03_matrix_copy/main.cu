#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__global__ void matrix_copy_kernel(const float *__restrict__ A, float *__restrict__ B, size_t N)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = N / 4;

    const float4 *a4 = reinterpret_cast<const float4 *>(A);
    float4 *b4 = reinterpret_cast<float4 *>(B);

    for (size_t i = idx; i < n4; i += stride)
    {
        b4[i] = __ldg(a4 + i); // 只读路径显式用 __ldg()
    }
    for (size_t i = n4 * 4 + idx; i < N; i += stride)
    {
        B[i] = __ldg(A + i);
    }
}

void solve(const float *d_A, float *d_B, size_t N)
{
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;

    matrix_copy_kernel<<<blocks, THREADS>>>(d_A, d_B, N);
}

float benchmark(const float *A, float *B, size_t N, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(A, B, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(A, B, N);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void matrix_copy_cpu(const float *A, float *B, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        B[i] = A[i];
    }
}

int main()
{
    // 创建数据
    const size_t N = 4096 * 2048;
    std::vector<float> A(static_cast<size_t>(N));
    for (size_t i = 0; i < A.size(); ++i)
    {
        A[i] = static_cast<float>(i % 17);
    }

    std::vector<float> ref(N);
    matrix_copy_cpu(A.data(), ref.data(), N);

    // 分配 GPU 内存
    float *d_A, *d_B;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);

    // 可行性测试
    solve(d_A, d_B, N);
    cudaDeviceSynchronize();
    std::vector<float> got(N);
    cudaMemcpy(got.data(), d_B, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (ref[i] != got[i])
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
    float average_time = benchmark(d_A, d_B, N, warmup_iters, bench_iters);
    double total_bytes = 2.0 * bytes;
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("矩阵元素数 %zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", N, average_time, bandwidth_gbps);

    cudaFree(d_A);
    cudaFree(d_B);
}
