#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define TILE 32

__global__ void matrix_transpose_float4(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];

    // blockDim = (8, 32)
    int tx = threadIdx.x; // 0~7
    int ty = threadIdx.y; // 0~31

    int x = blockIdx.x * TILE + tx * 4; // 全局列起点 (0, 4, 8...)
    int y = blockIdx.y * TILE + ty;     // 全局行起点

    // 阶段 1：读入 4 个连续的 float，存入共享内存
    if (x + 3 < cols && y < rows) {
        float4 v = *reinterpret_cast<const float4*>(&in[y * cols + x]);
        tile[ty][tx * 4 + 0] = v.x;
        tile[ty][tx * 4 + 1] = v.y;
        tile[ty][tx * 4 + 2] = v.z;
        tile[ty][tx * 4 + 3] = v.w;
    }
    __syncthreads();

    // 阶段 2：从共享内存转置读出，写回全局内存
    // 输出的行对应输入的列（乘 4 是因为转置后，列变成了连续的 x 跨度）
    int new_x = blockIdx.y * TILE + tx * 4; 
    int new_y = blockIdx.x * TILE + ty;

    if (new_x + 3 < rows && new_y < cols) {
        float4 v;
        // 这里从共享内存里读取 4 个相邻行的同一列，拼成一个 float4
        v.x = tile[tx * 4 + 0][ty];
        v.y = tile[tx * 4 + 1][ty];
        v.z = tile[tx * 4 + 2][ty];
        v.w = tile[tx * 4 + 3][ty];
        
        // 一次性写出 16 字节
        *reinterpret_cast<float4*>(&out[new_y * rows + new_x]) = v;
    }
}

void solve(const float *in, float *out, size_t rows, size_t cols)
{
    dim3 block(8, 32);
    dim3 grid(CEIL(cols, 32), CEIL(rows, 32));
    matrix_transpose_float4<<<grid, block>>>(in, out, rows, cols);
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