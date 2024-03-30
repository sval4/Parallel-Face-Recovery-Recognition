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

extern double *train_images;

extern double* test_images;
extern double* dist;

extern "C" void createData(size_t length, size_t cols, size_t test_length){
    cudaMallocManaged( &occluded_image, ( test_length * cols * sizeof(double))); 
    cudaMemset(occluded_image, 0, test_length * cols * sizeof(double));

    cudaMallocManaged( &occluded_mask, ( test_length * cols * sizeof(int))); 
    cudaMemset(occluded_mask, 0, test_length * cols * sizeof(int));

    cudaMallocManaged( &dist, ( length * cols *sizeof(double))); 
    cudaMemset(dist, 0, length * cols *sizeof(double));

    cudaMallocManaged( &train_images, ( length * cols * sizeof(double)));
    cudaMemset(train_images, 0, length * cols * sizeof(double));

    cudaMallocManaged( &test_images, ( test_length * cols * sizeof(double))); 
    cudaMemset(test_images, 0, test_length * cols * sizeof(double));
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

extern "C" void normalizeKernelLaunch(double** data, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    normalizeKernel<<<gridSize, blockSize>>>(*data, rows, cols);
    cudaDeviceSynchronize();
}


__global__ void meanCenterKernel(double* data, double* means, size_t rows, size_t cols){
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += blockDim.x * gridDim.x) {
        size_t loc = idx % cols;
        data[idx] -= means[loc];
    }
}


extern "C" void meanCenterKernelLaunch(double** data, double* means, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    meanCenterKernel<<<gridSize, blockSize>>>(*data, means, rows, cols);
    cudaDeviceSynchronize();
}


__global__ void norm2Kernel(double* images, double* test, size_t rows, size_t cols, size_t start_test, double* answer){
    size_t index;
    for(index = blockIdx.x * blockDim.x + threadIdx.x; index < rows; index += blockDim.x * gridDim.x) {
        double sum = 0.0;
        size_t start = index * cols;
        for(int i = start; i < cols + start; i++){
            sum += (images[i] - test[start_test + i - start]) * (images[i] - test[start_test + i - start]);
        }
        answer[index] = sqrt(sum);
    }
}

extern "C" void norm2KernelLaunch(double** data, double* test, size_t rows, size_t cols, size_t start, size_t threadsCount, double** answer){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    norm2Kernel<<<gridSize, blockSize>>>(*data, test, rows, cols, start, *answer);
    cudaDeviceSynchronize();
}