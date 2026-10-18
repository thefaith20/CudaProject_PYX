#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 多维网格映射法，但是写操作地址不连续，性能一般
__global__ void nchw_to_nhwc_native(const float *__restrict__ in, float *__restrict__ out, size_t N, size_t C, size_t H,
                                    size_t W)
{
    size_t w = blockDim.x * blockIdx.x + threadIdx.x;
    size_t h = blockDim.y * blockIdx.y + threadIdx.y;
    size_t nc = blockIdx.z;

    if (w < W && h < H && nc < N * C)
    {
        size_t n = nc / C;
        size_t c = nc % C;

        size_t in_idx = n * C * H * W + c * H * W + h * W + w;  // 读地址连续
        size_t out_idx = n * H * W * C + h * W * C + w * C + c; // 写地址非连续

        out[out_idx] = in[in_idx];
    }
}

void solve_native(const float *in, float *out, size_t N, size_t C, size_t H, size_t W)
{
    dim3 block(32, 8);
    dim3 grid(CEIL(W, 32), CEIL(H, 8), N * C);
    nchw_to_nhwc_native<<<grid, block>>>(in, out, N, C, H, W);
}

#define TILE 16

// 矩阵转置法，使用共享内存，读和写全都变成了完美的 128 字节合并访问，性能高
__global__ void nchw_to_nhwc_shared(const float *__restrict__ in, float *__restrict__ out, size_t N, size_t C, size_t H,
                                    size_t W)
{
    __shared__ float smem[TILE][TILE + 1];

    // 可以看作 [N][C][HW] 转置成 [N][HW][C]
    size_t HW = H * W;
    size_t n = blockIdx.z;

    // 读入
    size_t col_hw = blockDim.x * blockIdx.x + threadIdx.x; // 列对应 HW
    size_t row_c = blockDim.y * blockIdx.y + threadIdx.y;  // 行对应 C

    if (row_c < C && col_hw < HW)
    {
        smem[threadIdx.y][threadIdx.x] = in[n * HW * C + row_c * HW + col_hw];
    }
    __syncthreads();

    // 写出
    size_t row_hw = blockDim.x * blockIdx.x + threadIdx.y; // 转置，行对应 HW
    size_t col_c = blockDim.y * blockIdx.y + threadIdx.x;  // 转置，列对应 C

    if (row_hw < HW && col_c < C)
    {
        out[n * C * HW + row_hw * C + col_c] = smem[threadIdx.x][threadIdx.y];
    }
}

void solve_shared(const float *in, float *out, size_t N, size_t C, size_t H, size_t W)
{
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(H * W, TILE), CEIL(C, TILE), N);
    nchw_to_nhwc_shared<<<grid, block>>>(in, out, N, C, H, W);
}

template <typename LaunchFn>
float benchmark(const float *in, float *out, LaunchFn launch, size_t N, size_t C, size_t H, size_t W, int warmup_iters,
                int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        launch(in, out, N, C, H, W);
    }

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        launch(in, out, N, C, H, W);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void nchw_to_nhwc_cpu(const float *in, float *out, size_t N, size_t C, size_t H, size_t W)
{
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t hw = 0; hw < H * W; ++hw)
            {
                out[n * H * W * C + hw * C + c] = in[n * C * H * W + c * H * W + hw];
            }
        }
    }
}

int main()
{
    size_t N = 16, C = 64, H = 256, W = 256;
    size_t total = N * C * H * W;
    size_t bytes = total * sizeof(float);
    std::vector<float> in(total), ref(total), got_n(total), got_s(total);
    for (size_t i = 0; i < total; ++i)
    {
        in[i] = static_cast<float>(i % 100);
    }
    nchw_to_nhwc_cpu(in.data(), ref.data(), N, C, H, W);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);

    solve_native(d_in, d_out, N, C, H, W);
    cudaDeviceSynchronize();
    cudaMemcpy(got_n.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < total; ++i)
    {
        if (fabsf(ref[i] - got_n[i]) > 1e-5f * (1.f + fabsf(ref[i])))
        {
            printf("native 版本于 %zu 处出现错误：参考值 = %f, 实际值 = %f\n", i, ref[i], got_n[i]);
            break;
        }
    }

    solve_shared(d_in, d_out, N, C, H, W);
    cudaDeviceSynchronize();
    cudaMemcpy(got_s.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < total; ++i)
    {
        if (fabsf(ref[i] - got_s[i]) > 1e-5f * (1.f + fabsf(ref[i])))
        {
            printf("shared 版本于 %zu 处出现错误：参考值 = %f, 实际值 = %f\n", i, ref[i], got_s[i]);
            break;
        }
    }

    const int warmup_iters = 10;
    const int bench_iters = 100;
    double total_bytes = 2.0 * bytes;
    float average_time_native = benchmark(d_in, d_out, solve_native, N, C, H, W, warmup_iters, bench_iters);
    double bandwidth_gbps_native = total_bytes / (average_time_native * 1e-3) / 1e9;
    float average_time_shared = benchmark(d_in, d_out, solve_shared, N, C, H, W, warmup_iters, bench_iters);
    double bandwidth_gbps_shared = total_bytes / (average_time_shared * 1e-3) / 1e9;

    printf("矩阵维度 %zu*%zu*%zu*%zu\n", N, C, H, W);
    printf("native 版本: 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", average_time_native, bandwidth_gbps_native);
    printf("shared 版本: 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", average_time_shared, bandwidth_gbps_shared);

    cudaFree(d_in);
    cudaFree(d_out);
}