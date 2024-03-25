#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <curand_kernel.h>


extern double* occluded_image;
extern int* occluded_mask;

extern "C" void createData(size_t length, size_t cols){
    cudaMallocManaged( &occluded_image, ( length * cols * sizeof(double))); 
    cudaMemset(occluded_image, 0, length * cols * sizeof(double));

    cudaMallocManaged( &occluded_mask, ( length * cols * sizeof(int))); 
    cudaMemset(occluded_mask, 0, length * cols * sizeof(int));
}

__global__ void occludedKernel(double* data, double* test_images, int* occluded_mask, size_t rows, size_t cols) {
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cols * rows; idx += blockDim.x * gridDim.x) {
        size_t start = idx / cols;
        start *= cols;
        int width = sqrt((float) cols);
        if(occluded_mask[idx] == 1){
            int left = idx - 1;
            while(left >= start && left % width != (width - 1) && occluded_mask[left] == 1) left--;
            int right = idx + 1;
            while(right < start + width && right % width != 0 && occluded_mask[right] == 1) right++;
            int top = idx - width;
            while(top >= start && occluded_mask[top] == 1) top-= width;
            int bottom = idx + width;
            while(bottom < start + cols && occluded_mask[bottom] == 1) bottom+= width;

            double interpolated_value = 0.0;
            int count = 0;
            if(left >= 0 && occluded_mask[left] == 0){
                interpolated_value += test_images[left];
                count++;
            }
            if(right < cols && occluded_mask[right] == 0){
                interpolated_value += test_images[right];
                count++;
            }
            if(top >= 0 && occluded_mask[top] == 0){
                interpolated_value += test_images[top];
                count++;
            }
            if(bottom < cols && occluded_mask[bottom] == 0){
                interpolated_value += test_images[bottom];
                count++;
            }
            if(count > 0){
                data[idx] = interpolated_value / count;
            }
        }else{
            data[idx] = test_images[idx]; 
        }
    }
}

extern "C" void occludedKernelLaunch(double** data, double* test_images, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    occludedKernel<<<gridSize, blockSize>>>(*data, test_images, occluded_mask, rows, cols);
    cudaDeviceSynchronize();
}


__global__ void createMaskKernel(int* data, size_t rows, size_t cols){
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cols * rows; idx += blockDim.x * gridDim.x) {
        curandState state;
        curand_init(123, idx, 0, &state);
        //Performance varies slightly because of random number generation
        data[idx] = (curand_uniform(&state) < 0.5) ? 0 : 1;
    }
}

extern "C" void createMask(int** data, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    createMaskKernel<<<gridSize, blockSize>>>(*data, rows, cols);
    cudaDeviceSynchronize();
}

extern "C" void copyToCPUOccluded(double* data, size_t length, int start){
    cudaMemcpy(data, occluded_image + (start * length), length * sizeof(double), cudaMemcpyDeviceToHost);
}