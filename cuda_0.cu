#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

void check_result(float *host_ref, float *dev_ref, const int N)
{
    double var = 1e-8;
    int flag = 1;
    for (int i = 0; i < N; ++i)
    {
        if (abs(host_ref[i] - dev_ref[i]) > var)
        {
            flag = 0;
            break;
        }
    }
    if (flag == 0)
        printf("计算误差过大\n");
    else
        printf("误差匹配\n");
}

// compute on cpu
void tensor_add_cpu(float *a, float *b, float *res, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        res[i] = a[i] + b[i];
    }
}

// compute on gpu
__global__ void tensor_add_gpu(float *a, float *b, float *res, const int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    res[idx] = a[idx] + b[idx];
}

void initialize_data(float *a, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = i;
    }
}

int main()
{
    cudaSetDevice(0);

    int N = 1024;
    size_t nBytes = N * sizeof(float);

    float *h_a, *h_b, *h_res, *ground_truth;
    float *d_a, *d_b, *d_res;

    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_res = (float *)malloc(nBytes);
    ground_truth = (float *)malloc(nBytes);
    initialize_data(h_a, N);
    initialize_data(h_b, N);

    // malloc device global memory
    cudaMalloc((float **)(&d_a), nBytes);
    cudaMalloc((float **)(&d_b), nBytes);
    cudaMalloc((float **)(&d_res), nBytes);

    // copy memory from cpu to gpu
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // initialize grid block size
    dim3 block(32);
    dim3 grid(N / 32);

    // computing on gpu
    tensor_add_gpu<<<grid, block>>>(d_a, d_b, d_res, N);
    printf("开始在GPU上计算...\n");

    // computing on cpu
    tensor_add_cpu(h_a, h_b, ground_truth, N);

    // copy result memory from gpu to cpu
    cudaMemcpy(h_res, d_res, nBytes, cudaMemcpyDeviceToHost);

    check_result(ground_truth, h_res, N);

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    // free cpu memory
    free(h_a);
    free(h_b);
    free(h_res);
    free(ground_truth);
}