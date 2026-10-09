#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

// C = {A[0], B[0], A[1], B[1]...}
__global__ void interleave_arrays_kernel(const float *__restrict__ A, const float *__restrict__ B,
                                         float *__restrict__ C, size_t N)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n2 = N / 2;

    const float2 *a2 = reinterpret_cast<const float2 *>(A);
    const float2 *b2 = reinterpret_cast<const float2 *>(B);
    float4 *c4 = reinterpret_cast<float4 *>(C);

    for (size_t i = idx; i < n2; i += stride)
    {
        c4[i] = make_float4(a2[i].x, b2[i].x, a2[i].y, b2[i].y);
    }

    float2 *c2 = reinterpret_cast<float2 *>(C);
    for (size_t i = n2 * 2 + idx; i < N; i += stride)
    {
        c2[i] = make_float2(A[i], B[i]);
    }
}

void solve(const float *A, const float *B, float *C, size_t N)
{
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    interleave_arrays_kernel<<<blocks, THREADS>>>(A, B, C, N);
}

float benchmark(const float *A, const float *B, float *C, size_t N, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(A, B, C, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(A, B, C, N);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void interleave_arrays_cpu(const float *A, const float *B, float *C, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        float2 *c = reinterpret_cast<float2 *>(C);
        c[i] = make_float2(A[i], B[i]);
    }
}

int main()
{
    const size_t N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    std::vector<float> A(N), B(N), ref(N * 2), got(N * 2);
    for (size_t i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>((i % 13) - 6);
        B[i] = static_cast<float>((i % 17) - 8);
    }

    interleave_arrays_cpu(A.data(), B.data(), ref.data(), N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes * 2);
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), d_C, bytes * 2, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N * 2; ++i)
    {
        if (fabsf(ref[i] - got[i]) > 1e-5f * (1.f + fabsf(ref[i])))
        {
            printf("于 %zu 处出现错误：参考值 = %f, 实际值 = %f\n", i, ref[i], got[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "成功\n" : "错误\n");

    const int warmup_iters = 10;
    const int bench_iters = 100;
    float average_time = benchmark(d_A, d_B, d_C, N, warmup_iters, bench_iters);
    double total_bytes = 4.0 * N * sizeof(float);
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("N=%zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", N, average_time, bandwidth_gbps);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}