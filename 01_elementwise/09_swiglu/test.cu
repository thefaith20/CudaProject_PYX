#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float swiglu(float a, float b) { return a / (1.f + __expf(-a)) * b; }
__device__ __forceinline__ float4 swiglu4(float4 a, float4 b)
{
    return make_float4(swiglu(a.x, b.x), swiglu(a.y, b.y), swiglu(a.z, b.z), swiglu(a.w, b.w));
}

__global__ void swiglu_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, size_t N)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x; // stride 网格跨步循环，可让线程自适应处理更多数据
    const size_t n = N / 4;

    const float4 *a4 = reinterpret_cast<const float4 *>(A);
    const float4 *b4 = reinterpret_cast<const float4 *>(B);
    float4 *c4 = reinterpret_cast<float4 *>(C);

    for (size_t i = idx; i < n; i += stride)
    {
        c4[i] = swiglu4(a4[i], b4[i]);
    }
    for (size_t i = n * 4 + idx; i < N; i += stride)
    {
        C[i] = swiglu(A[i], B[i]);
    }
}

void solve(const float *A, const float *B, float *C, size_t N)
{
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    swiglu_kernel<<<blocks, THREADS>>>(A, B, C, N);
}

float benchmark(const float *d_A, const float *d_B, float *d_C, size_t N, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热跑几次让 GPU 进入稳定状态
    for (int i = 0; i < warmup_iters; ++i)
        solve(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // 阻塞直至之前发出的 CUDA 指令完成

    // 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
        solve(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void swiglu_cpu(const float *A, const float *B, float *C, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        C[i] = A[i] / (1.f + expf(-A[i])) * B[i];
    }
}

int main()
{
    // 创建数据
    const size_t N = 1 << 24;
    size_t bytes = N * sizeof(float);
    std::vector<float> A(N), B(N), ref(N), got(N);
    for (size_t i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>((i % 13) - 6);
    }
    for (size_t i = 0; i < N; ++i)
    {
        B[i] = static_cast<float>((i % 17) - 8);
    }

    swiglu_cpu(A.data(), B.data(), ref.data(), N);

    // 分配 GPU 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // 可行性验证
    solve(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (fabsf(ref[i] - got[i]) > 1e-5f * (1.f + fabsf(ref[i])))
        {
            printf("于 %zu 处出现错误：参考值 = %f, 实际值 = %f\n", i, ref[i], got[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "成功\n" : "错误\n");

    // 性能测试
    const int warmup_iters = 10;
    const int bench_iters = 100;
    float average_time = benchmark(d_A, d_B, d_C, N, warmup_iters, bench_iters);
    double total_bytes = 3.0 * N * sizeof(float);                      // 每次迭代传输 3*N*4 字节(A读，B读，C写)
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9; // 有效带宽

    printf("N=%zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", N, average_time, bandwidth_gbps);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
