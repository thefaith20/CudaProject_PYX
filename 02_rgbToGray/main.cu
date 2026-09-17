#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 输入图像在显存中是按 uchar3 (RGB三通道) 紧密排列的
__global__ void rgbToGray(const uchar3 *d_img, unsigned char *d_gray, int width, int height)
{
    // 计算全局列和行
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (x < width && y < height)
    {
        // 二维转一维索引
        int idx = y * width + x;

        // 读取原始RGB像素
        uchar3 pixel = d_img[idx];

        // 计算灰度值 (固定加权公式) Y = 0.299R + 0.587G + 0.114B
        unsigned char gray = static_cast<unsigned char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);

        // 写回显存
        d_gray[idx] = gray;
    }
}

using namespace std;
int main()
{
    string imagePath = "inputImg.png";
    cv::Mat img = cv::imread(imagePath);

    if (img.empty())
    {
        cerr << "无法加载图像：" << imagePath << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    uchar *d_img;          // GPU上的原始图像数据
    unsigned char *d_gray; // GPU上的灰度图像
    cout << "图像尺寸：" << width << " x " << height << ", 通道数：" << channels << endl;

    // 定义Cuda内核的块和网格大小
    dim3 blockSize(32, 8);
    const int interations = 10000;

    size_t imgSize = width * height * sizeof(uchar3);
    size_t graySize = width * height * sizeof(unsigned char);

    // 在GPU上分配内存
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_gray, graySize);

    // 将图像数据从CPU复制到GPU
    cudaMemcpy(d_img, img.data, imgSize, cudaMemcpyHostToDevice);

    // 计算网格大小：用图像尺寸除以块大小并向上取整
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 启动CUDA内核进行RGB到灰度的转换
    rgbToGray<<<gridSize, blockSize>>>(reinterpret_cast<uchar3 *>(d_img), d_gray, width, height);
    // 检查内核启动是否成功
    cudaGetLastError();
    // 同步
    cudaDeviceSynchronize();

    // 测试时间
    cudaEvent_t start, stop;
    float total_time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "正在进行GPU性能测试..." << endl;

    for (int i = 0; i < interations; i++)
    {
        cudaEventRecord(start);
        rgbToGray<<<gridSize, blockSize>>>(reinterpret_cast<uchar3 *>(d_img), d_gray, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float single_time = 0.0f;
        cudaEventElapsedTime(&single_time, start, stop);
        total_time += single_time;
    }

    float avg_time = total_time / interations;
    cout << "平均每次转换时间: " << avg_time << " ms" << endl;

    // 将结果从GPU复制回CPU
    cv::Mat grayImg(height, width, CV_8UC1);
    cudaMemcpy(grayImg.data, d_gray, graySize, cudaMemcpyDeviceToHost);

    // 保存灰度图像
    cv::imwrite("outputImg.png", grayImg);

    // 释放GPU内存
    cudaFree(d_img);
    cudaFree(d_gray);

    return 0;
}