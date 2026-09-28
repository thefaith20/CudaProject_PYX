#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define MAX_BLOCKS 1024
#define CEIL(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ float rgb_to_gray(float3 p) { return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z; }

__global__ void rgb_to_gray_kernel(const float *__restrict__ in, float *__restrict__ out, size_t nPix)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    const float3 *in3 = reinterpret_cast<const float3 *>(in);

    for (size_t i = idx; i < nPix; i += stride)
    {
        out[i] = rgb_to_gray(in3[i]);
    }
}

void solve(const float *in, float *out, size_t nPix)
{
    int blocks = CEIL(nPix, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    rgb_to_gray_kernel<<<blocks, THREADS>>>(in, out, nPix);
}

float benchmark(const float *in, float *out, size_t nPix, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
        solve(in, out, nPix);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
        solve(in, out, nPix);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void rgb_to_gray_cpu(const float *in, float *out, size_t nPix)
{
    for (size_t i = 0; i < nPix; ++i)
    {
        out[i] = 0.299f * in[i * 3 + 0] + 0.587f * in[i * 3 + 1] + 0.114f * in[i * 3 + 2];
    }
}

int main()
{
    size_t width = 8192, height = 4096;
    size_t nPix = width * height;
    std::vector<float> in(nPix * 3), ref(nPix), got(nPix);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<float>(i % 256);

    rgb_to_gray_cpu(in.data(), ref.data(), nPix);

    size_t bytes_in = nPix * 3 * sizeof(float);
    size_t bytes_out = nPix * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes_in);
    cudaMalloc(&d_out, bytes_out);
    cudaMemcpy(d_in, in.data(), bytes_in, cudaMemcpyHostToDevice);

    solve(d_in, d_out, nPix);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), d_out, bytes_out, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < nPix; ++i)
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
    float average_time = benchmark(d_in, d_out, nPix, warmup_iters, bench_iters);
    double total_bytes = 3 * nPix * sizeof(float) + nPix * sizeof(float);
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("width = %zu, height = %zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", width, height, average_time,
           bandwidth_gbps);

    cudaFree(d_in);
    cudaFree(d_out);
}