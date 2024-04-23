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

extern double *d_means;

extern double* test_images;
extern double* dist;

extern "C" void createData(size_t length, size_t cols, size_t test_length){
    cudaMallocManaged( &occluded_image, ( test_length * cols * sizeof(double))); 
    cudaMemset(occluded_image, 0, test_length * cols * sizeof(double));

    cudaMallocManaged( &occluded_mask, ( test_length * cols * sizeof(int))); 
    cudaMemset(occluded_mask, 0, test_length * cols * sizeof(int));

    cudaMallocManaged( &dist, ( length * cols *sizeof(double))); 
    cudaMemset(dist, 0, length * cols *sizeof(double));

    cudaMallocManaged( &d_means, ( cols *sizeof(double))); 
    cudaMemset(d_means, 0, cols *sizeof(double));

    cudaMallocManaged( &train_images, ( length * cols * sizeof(double)));
    cudaMemset(train_images, 0, length * cols * sizeof(double));

    cudaMallocManaged( &test_images, ( test_length * cols * sizeof(double))); 
    cudaMemset(test_images, 0, test_length * cols * sizeof(double));
}

__global__ void occludedKernel(double* data, double* test_images, int* occluded_mask_arg, size_t start_row, size_t rows, size_t cols) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < cols * rows; index += blockDim.x * gridDim.x) {
        size_t idx = index + start_row * cols;
        size_t start = (int) idx / cols;
        start *= cols;
        int width = sqrt((float) cols);
        if(occluded_mask_arg[idx] == 1){
            int left = idx - 1;
            while(left > start && left % width != (width - 1) && occluded_mask_arg[left] == 1) left--;
            int right = idx + 1;
            while(right < start + width && right % width != 0 && occluded_mask_arg[right] == 1) right++;
            int top = idx - width;
            while(top > start && occluded_mask_arg[top] == 1) top-= width;
            int bottom = idx + width;
            while(bottom < start + cols && occluded_mask_arg[bottom] == 1) bottom+= width;

            double interpolated_value = 0.0;
            int count = 0;
            if(left >= start && left % width != (width - 1) && occluded_mask_arg[left] == 0){
                interpolated_value += test_images[left];
                count++;
            }
            if(right < start + width && right % width != 0 && occluded_mask_arg[right] == 0){
                interpolated_value += test_images[right];
                count++;
            }
            if(top >= start && occluded_mask_arg[top] == 0){
                interpolated_value += test_images[top];
                count++;
            }
            if(bottom < start + cols && occluded_mask_arg[bottom] == 0){
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

extern "C" void occludedKernelLaunch(double** data, double* test_images, int* occluded_mask_arg, size_t start, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    occludedKernel<<<gridSize, blockSize>>>(*data, test_images, occluded_mask_arg, start, rows, cols);
    cudaDeviceSynchronize();
}


__global__ void createMaskKernel(int* data, size_t start, size_t rows, size_t cols){
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < cols * rows; index += blockDim.x * gridDim.x) {
        size_t idx = index + start * cols;
        curandState state;
        curand_init(123, idx, 0, &state);
        //Performance won't vary, if same test size is used
        data[idx] = (curand_uniform(&state) < 0.5) ? 0 : 1;
    }
}

extern "C" void createMask(int** data, size_t start, size_t rows, size_t cols, size_t threadsCount){
    size_t blockSize = threadsCount;
    size_t gridSize = (((rows * cols) + blockSize - 1) / blockSize);
    createMaskKernel<<<gridSize, blockSize>>>(*data, start, rows, cols);
    cudaDeviceSynchronize();
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
    // cudaMemcpy(d_means, *means, cols*sizeof(double), cudaMemcpyHostToDevice);
    meanCenterKernel<<<gridSize, blockSize>>>(*data, means, rows, cols);
    cudaDeviceSynchronize();
}

extern "C" void freeData(){
    cudaFree(occluded_image);
    cudaFree(occluded_mask);
    cudaFree(train_images);
    cudaFree(test_images);
    cudaFree(dist);
    cudaFree(d_means);
}