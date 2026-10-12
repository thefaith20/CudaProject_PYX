#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define MAX_BLOCKS 1024
#define CEIL(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ uchar4 invertColor(uchar4 v) { return make_uchar4(255 - v.x, 255 - v.y, 255 - v.z, v.w); }

// char4 版本
__global__ void color_invert_kernel_v1(unsigned char *__restrict__ image, int nPix)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    uchar4 *p = reinterpret_cast<uchar4 *>(image);

    for (size_t i = idx; i < nPix; i += stride)
    {
        p[i] = invertColor(p[i]);
    }
}

// char4×4 版本
__global__ void color_invert_kernel_v2(unsigned char *__restrict__ image, int nPix)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = nPix / 4;

    uint4 *p = reinterpret_cast<uint4 *>(image);

    for (size_t i = idx; i < n4; i += stride)
    {
        uint4 v = p[i]; // 128 位加载
        uchar4 *px = reinterpret_cast<uchar4 *>(&v);
        px[0] = invertColor(px[0]);
        px[1] = invertColor(px[1]);
        px[2] = invertColor(px[2]);
        px[3] = invertColor(px[3]);
        p[i] = v; // 128 位写回
    }

    // 处理尾部剩余不足 4 个像素的部分
    for (size_t i = n4 * 4 + idx; i < (size_t)nPix; i += stride)
    {
        uchar4 *p4 = reinterpret_cast<uchar4 *>(image);
        p4[i] = invertColor(p4[i]);
    }
}

void solve(unsigned char *image, int width, int height)
{
    const int nPix = width * height;
    const size_t n4 = nPix / 4;
    int blocks = CEIL(n4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;

    color_invert_kernel_v2<<<blocks, THREADS>>>(image, nPix);
}

float benchmark(unsigned char *d_p, int width, int height, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(d_p, width, height);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(d_p, width, height);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void invert_color_cpu(unsigned char *image, int width, int height)
{
    int nPix = width * height;
    for (int i = 0; i < nPix; ++i)
    {
        image[i * 4 + 0] = 255 - image[i * 4 + 0];
        image[i * 4 + 1] = 255 - image[i * 4 + 1];
        image[i * 4 + 2] = 255 - image[i * 4 + 2];
    }
}

int main()
{
    const int width = 8192;
    const int height = 4096;
    const int n = width * height * 4;
    std::vector<unsigned char> image(n);
    for (int i = 0; i < n; ++i)
    {
        image[i] = static_cast<unsigned char>(i * 17);
    }

    unsigned char *d_image = nullptr;
    size_t bytes = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_image, bytes);

    // 正确性验证
    std::vector<unsigned char> ref = image;
    invert_color_cpu(ref.data(), width, height);

    cudaMemcpy(d_image, image.data(), bytes, cudaMemcpyHostToDevice);
    solve(d_image, width, height);
    cudaDeviceSynchronize();

    std::vector<unsigned char> got(n);
    cudaMemcpy(got.data(), d_image, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < n; ++i)
    {
        if (got[i] != ref[i])
        {
            printf("FAIL  [%d]  got=%d, ref=%d\n", i, got[i], ref[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "OK  color\n" : "FAIL color\n");

    // 性能验证
    cudaMemcpy(d_image, image.data(), bytes, cudaMemcpyHostToDevice);

    const int warmup_iters = 10;
    const int bench_iters = 100;
    float average_time = benchmark(d_image, width, height, warmup_iters, bench_iters);
    double total_bytes = 2.0 * bytes;
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("width = %d, height = %d, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", width, height, average_time,
           bandwidth_gbps);

    cudaFree(d_image);
}