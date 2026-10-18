#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define TILE 16 // 每个 block 处理的矩阵分块大小

// =======================================================================================
// 朴素版：未使用共享内存，总会有地址非合并，性能低
// 读：in[row * cols + col]，col 变化时地址连续 → 合并  row 变化时地址跳 cols 个元素 → 非合并
// 写：out[col * rows + row]，col 变化时地址跳 rows 个元素 → 非合并  row 变化时地址连续 → 合并
// =======================================================================================
__global__ void matrix_transpose_naive(const float *in, float *out, size_t rows, size_t cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
    {
        out[col * rows + row] = in[row * cols + col];
    }
}

// ================================================================
// 正式版，使用共享内存，并且避免了 shared memory 内部的 bank conflict
// ================================================================
__global__ void matrix_transpose_shared(const float *in, float *out, size_t rows, size_t cols)
{
    __shared__ float smem[TILE][TILE + 1]; // TILE + 1

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        smem[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    size_t trans_x = blockDim.y * blockIdx.y + threadIdx.x; // 转置 x
    size_t trans_y = blockDim.x * blockIdx.x + threadIdx.y; // 转置 y
    if (trans_x < rows && trans_y < cols)
    {
        // threadIdx.x 连续变化时，对应 smem 内地址 0,33,66... 避免了 bank conflict
        out[trans_y * rows + trans_x] = smem[threadIdx.x][threadIdx.y];
    }
}

void solve(const float *in, float *out, size_t rows, size_t cols)
{
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(cols, TILE), CEIL(rows, TILE));
    matrix_transpose_shared<<<grid, block>>>(in, out, rows, cols);
}

float benchmark(const float *in, float *out, size_t rows, size_t cols, int warmup_iters, int bench_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; ++i)
    {
        solve(in, out, rows, cols);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
        solve(in, out, rows, cols);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float total_time = 0.f;
    cudaEventElapsedTime(&total_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / bench_iters;
}

void matrix_transpose_cpu(const float *in, float *out, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

int main()
{
    const size_t rows = 8192, cols = 4096;
    size_t bytes = rows * cols * sizeof(float);
    std::vector<float> in(rows * cols), ref(rows * cols), got(rows * cols);
    for (size_t i = 0; i < in.size(); ++i)
    {
        in[i] = static_cast<float>(i % 13);
    }
    matrix_transpose_cpu(in.data(), ref.data(), rows, cols);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);

    solve(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < rows * cols; ++i)
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
    float average_time = benchmark(d_in, d_out, rows, cols, warmup_iters, bench_iters);
    double total_bytes = 2.0 * bytes;
    double bandwidth_gbps = total_bytes / (average_time * 1e-3) / 1e9;

    printf("矩阵维度 %zu*%zu, 平均耗时 %.4f ms, 有效带宽 %.1f GB/s\n", cols, rows, average_time, bandwidth_gbps);

    cudaFree(d_in);
    cudaFree(d_out);
}