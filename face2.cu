#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

extern double *train_images;

extern double* test_images;

extern double* result;

extern double* occluded_image;

extern int* train_labels;  
extern int* test_labels;


extern "C" void createData(size_t length, size_t cols){
    cudaMallocManaged( &train_images, ( length * cols * sizeof(double)));
    cudaMemset(train_images, 0, length * cols * sizeof(double));

    cudaMallocManaged( &test_images, ( 40 * cols * sizeof(double))); 
    cudaMemset(test_images, 0, 40 * cols * sizeof(double));

    cudaMallocManaged( &result, ( length * sizeof(double))); 
    cudaMemset(result, 0, length * sizeof(double));

    cudaMallocManaged( &occluded_image, ( cols * sizeof(double))); 
    cudaMemset(occluded_image, 0, cols * sizeof(double));
}

extern "C" void copyToDeviceTrainImages(double* data, size_t length){
    cudaMemcpy(train_images, data, length * sizeof(double), cudaMemcpyHostToDevice);
}

extern "C" void copyToDeviceTestImages(double* data, size_t length){
    cudaMemcpy(test_images, data, length * sizeof(double), cudaMemcpyHostToDevice);
}

extern "C" void copyToCPUResult(double* data, size_t length){
    cudaMemcpy(data, result, length * sizeof(double), cudaMemcpyDeviceToHost);
}

extern "C" void copyToCPUTest(double* data, size_t start, size_t length){
    cudaMemcpy(data, test_images + start, length * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void normalizeKernel(double* data, size_t rows, size_t cols){
    for(size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x; row_idx < rows; row_idx += blockDim.x * gridDim.x) {
        int start = row_idx * cols;
        double sum = 0.0;

        // Calculate the sum of squares of elements in the row
        for (int i = start; i < start + cols; i++) {
            sum += data[i] * data[i];
        }

        // Normalize each element in the row by dividing by the square root of the sum of squares
        double norm_factor = sqrt(sum);
        for (int i = start; i < start + cols; i++) {
            data[i] /= norm_factor;
        }
    }
}

__global__ void meanCenterKernel(double* data, double* means, size_t rows, size_t cols){
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += blockDim.x * gridDim.x) {
        size_t loc = idx % cols;
        data[idx] -= means[loc];
    }
}

__global__ void norm2Kernel(double* images, size_t rows, size_t cols, size_t start_test, double* answer, double* test_imgs){
    size_t index;
    for(index = blockIdx.x * blockDim.x + threadIdx.x; index < rows; index += blockDim.x * gridDim.x) {
        double sum = 0.0;
        size_t start = index * cols;
        for(int i = start; i < cols + start; i++){
            sum += (images[i] - test_imgs[start_test + i - start]) * (images[i] - test_imgs[start_test + i - start]);
        }
        answer[index] = sqrt(sum);
    }
}

extern "C" void normalizeKernelLaunch(double** data, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    normalizeKernel<<<gridSize, blockSize>>>(*data, rows, cols);
}

extern "C" void meanCenterKernelLaunch(double** data, double* means, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    meanCenterKernel<<<gridSize, blockSize>>>(*data, means, rows, cols);
}

extern "C" void norm2KernelLaunch(double** data, size_t rows, size_t cols, size_t start, size_t threadsCount, double** answer, int state){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    if(state == -1){
        norm2Kernel<<<gridSize, blockSize>>>(*data, rows, cols, 0, *answer, occluded_image);
    }else{
        norm2Kernel<<<gridSize, blockSize>>>(*data, rows, cols, start, *answer, test_images);
    }
    
}

__global__ void occluedKernel(double* data, int* mask, size_t rows, size_t cols) {
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cols; idx += blockDim.x * gridDim.x) {
        if(mask[idx] == 1){
            int left = idx - 1;
            while(left >= 0 && left % cols != (cols - 1) && mask[left] == 1) left--;
            int right = idx + 1;
            while(right < cols && right % cols != 0 && mask[right] == 1) right++;
            int top = idx - cols;
            while(top >= 0 && mask[top] == 1) top-= cols;
            int bottom = idx + cols;
            while(bottom < cols && mask[bottom] == 1) bottom+= cols;

            double interpolated_value = 0.0;
            int count = 0;
            if(left >= 0 && mask[left] == 0){
                interpolated_value += data[left];
                count++;
            }
            if(right < cols && mask[right] == 0){
                interpolated_value += data[right];
                count++;
            }
            if(top >= 0 && mask[top] == 0){
                interpolated_value += data[top];
                count++;
            }
            if(bottom < cols && mask[bottom] == 0){
                interpolated_value += data[bottom];
                count++;
            }
            if(count > 0){
                data[idx] = interpolated_value / count;
            }
        }
    }
}

extern "C" void occluedKernelLaunch(double** data, int* mask, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    occluedKernel<<<gridSize, blockSize>>>(*data, mask, rows, cols);
}