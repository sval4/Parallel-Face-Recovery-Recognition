#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <mpi.h>

#define MAX_LINE_LENGTH (4097 * 3 + 4096)
#define NUM_ROWS 360
#define NUM_COLS 4096

double *train_images = NULL;

double *result = NULL;

double* test_images = NULL;

double* occluded_image = NULL;

extern void createData(size_t length, size_t cols);
extern void copyToDeviceTrainImages(double* data, size_t length);
extern void copyToDeviceTestImages(double* data, size_t length);
extern void normalizeKernelLaunch(double** data, size_t rows, size_t cols, size_t threadsCount);
extern void meanCenterKernelLaunch(double** data, double* means, size_t rows, size_t cols, size_t threadsCount);
extern void norm2KernelLaunch(double** data, size_t rows, size_t cols, size_t start, size_t threadsCount, double** answer, int state);
extern void copyToCPUResult(double* data, size_t length);
extern void copyToCPUTest(double* data, size_t start, size_t length);
extern void occluedKernelLaunch(double** data, int* mask, size_t rows, size_t cols, size_t threadsCount);

void colMeans(double* images, double* means, int rows_per_rank){
    int i, j;
    double local_sum[NUM_COLS] = {0};

    for (j = 0; j < NUM_COLS; j++) {
        for (i = 0; i < rows_per_rank; i++) {
            local_sum[j] += images[i * NUM_COLS + j];
        }
        MPI_Allreduce(MPI_IN_PLACE, &local_sum[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        means[j] = local_sum[j] / NUM_ROWS;
    }
}

int matchImage(double** images, int rows_per_rank, size_t threads_per_block){
    int i;
    double minDist = DBL_MAX;
    int index = 0;
    double* check = calloc(rows_per_rank, sizeof(double));
    copyToCPUResult(check, rows_per_rank);

    for(i = 0; i < rows_per_rank; i++){
        if(minDist > check[i]){
            minDist = check[i];
            index = i;
        }
    }
    double global_minDist;
    MPI_Allreduce(&minDist, &global_minDist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    // if(myrank == 0){printf("GlobMinDist:%lf\n", global_minDist);}
    if (minDist > global_minDist) {
        index = -1;
    }
    free(check);
    return index;
}

// double norm2(double* images, int start, double* refImage){
//     int i;
//     double sum = 0;
//     for(i = start; i < NUM_COLS + start; i++){
//         sum += (images[i] - refImage[i - start]) * (images[i] - refImage[i - start]);
//     }
//     return sqrt(sum);

// }

// int matchImage(double* images, double* refImage, int rows_per_rank){
//     int i;
//     double minDist = DBL_MAX;
//     int index = 0;
//     for(i = 0; i < rows_per_rank; i++){
//         double dist = norm2(images, i * NUM_COLS, refImage);
//         if(minDist > dist){
//             minDist = dist;
//             index = i;
//         }
//     }
//     double global_minDist;
//     MPI_Allreduce(&minDist, &global_minDist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
//     // if(myrank == 0){printf("GlobMinDist:%lf\n", global_minDist);}
//     if (minDist > global_minDist) {
//         index = -1;
//     }
//     return index;
// }

int main(int argc, char *argv[]) {
    srand(123);
    int myrank, numranks;
    double start_time = 0;
    MPI_File file;
    MPI_Status status;
    char line[MAX_LINE_LENGTH + 2];
    char *token;
    size_t threads_per_block = 0;
    double* cpu_train_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* cpu_test_images = calloc(40 * NUM_COLS, sizeof(double));
    double* cpu_org_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* means = calloc(NUM_COLS, sizeof(double));
    int* train_labels = calloc(NUM_ROWS, sizeof(int));
    int* test_labels = calloc(40, sizeof(int));
    int field_count;
    int i,j;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    threads_per_block = atoi(argv[1]);

    int rows_per_rank = NUM_ROWS / numranks;
    int start_row = myrank * rows_per_rank;

    // +2 to handle the \r\n at end of each line
    MPI_Offset offset = start_row * (MAX_LINE_LENGTH + 2);

    if (myrank == 0) {
        start_time = MPI_Wtime();
    }

    MPI_File_open(MPI_COMM_WORLD, "faces_train360.csv", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    if (file == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        return 1;
    }
    createData(rows_per_rank, NUM_COLS);
//----------------------------------------------------------------------------------------------

    for (i = 0; i < rows_per_rank; i++) {
        MPI_File_seek(file, offset + i * (MAX_LINE_LENGTH + 2), MPI_SEEK_SET);
        MPI_File_read(file, line, (MAX_LINE_LENGTH + 2), MPI_CHAR, &status);
        line[MAX_LINE_LENGTH + 1] = '\0'; // Null-terminate the string
        char* token = strtok(line, ",");
        train_labels[i] = atoi(token);
        // if(myrank == 0){printf("train Label: %d\n",train_labels[i] );}
        int count = 0;
        token = strtok(NULL, ",");
        int field_count = 0;
        while (token != NULL) {
            cpu_org_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            cpu_train_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }
        // if(myrank == 0){printf("Count: %d\n", field_count);}
    }
    
    MPI_File_close(&file);
    copyToDeviceTrainImages(cpu_train_images, rows_per_rank * NUM_COLS);

    colMeans(cpu_train_images, means, rows_per_rank);
    meanCenterKernelLaunch(&train_images, means, rows_per_rank, NUM_COLS, threads_per_block);
    normalizeKernelLaunch(&train_images, rows_per_rank, NUM_COLS, threads_per_block);

//----------------------------------------------------------------------------------------------

    int test_rows_per_rank = 40 / numranks;
    int test_start_row = myrank * test_rows_per_rank;

    // +2 to handle the \r\n
    offset = test_start_row * (MAX_LINE_LENGTH + 2);

    MPI_File_open(MPI_COMM_WORLD, "faces_test.csv", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    if (file == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        return 1;
    }
    for (i = test_start_row; i < test_rows_per_rank + test_start_row; i++) {
        MPI_File_seek(file, offset + (i - test_start_row) * (MAX_LINE_LENGTH + 2), MPI_SEEK_SET);
        MPI_File_read(file, line, (MAX_LINE_LENGTH + 2), MPI_CHAR, &status);
        line[MAX_LINE_LENGTH + 1] = '\0'; // Null-terminate the string
        char* token = strtok(line, ",");
        test_labels[i] = atoi(token);
        // if(myrank == 1){printf("test Label: %d\n",test_labels[i]);}
        int count = 0;
        token = strtok(NULL, ",");
        int field_count = 0;
        while (token != NULL) {
            cpu_test_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }
        // if(myrank == 0){printf("Count: %d\n", field_count);}
    }
    MPI_File_close(&file);
    // See if you can optimize here by just making rank 0 normalize and meancenter
    MPI_Allgather(cpu_test_images + (test_start_row * NUM_COLS), test_rows_per_rank * NUM_COLS, MPI_DOUBLE, cpu_test_images, test_rows_per_rank * NUM_COLS, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(test_labels + test_start_row, test_rows_per_rank, MPI_INT, test_labels, test_rows_per_rank, MPI_INT, MPI_COMM_WORLD);
    
    copyToDeviceTestImages(cpu_test_images, 40 * NUM_COLS);

    meanCenterKernelLaunch(&test_images, means, 40, NUM_COLS, threads_per_block);
    normalizeKernelLaunch(&test_images, 40, NUM_COLS, threads_per_block);

//----------------------------------------------------------------------------------------------
    int correctCount = 0;
    for(i = 0; i < 40; i++){
        norm2KernelLaunch(&train_images, rows_per_rank, NUM_COLS, i * NUM_COLS, threads_per_block, &result, 1);
        int index = matchImage(&train_images, rows_per_rank, threads_per_block);
        if(index != -1 && train_labels[index] == test_labels[i]){
            correctCount++;
        }
    }
    int global_correctCount;
    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate: %lf%%\n", 100 * (double) global_correctCount/40);}

    correctCount = 0;
    int index = 0;
    for(j = 0; j < 40; j++){
        int* occlusion_mask = calloc(NUM_COLS, sizeof(int));
        if(myrank == 0){
            for(i = 0; i < NUM_COLS; i++){
                occlusion_mask[i] = rand() % 2;
            }
            for(i = 0; i < NUM_COLS; i++){
                occluded_image[i] = test_images[j * NUM_COLS + i];
            }
            occluedKernelLaunch(&occluded_image, occlusion_mask, rows_per_rank, NUM_COLS, threads_per_block);
        }
        MPI_Bcast(occluded_image, NUM_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        norm2KernelLaunch(&train_images, rows_per_rank, NUM_COLS, 0, threads_per_block, &result, -1);
        index = matchImage(&train_images, rows_per_rank, threads_per_block);
        if(index != -1 && train_labels[index] == test_labels[j]){
            correctCount++;
        }
        free(occlusion_mask);
    }

    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate Occlusion: %lf%%\n", 100 * (double) global_correctCount/40);}

    if (myrank == 0) {
        double t2 = MPI_Wtime();
        printf("Time is: %lf seconds\n", t2 - start_time);
    }

//     free(train_images);
//     free(test_images);
//     free(org_images);
//     free(means);
//     free(train_labels);
//     free(test_labels);

    MPI_Finalize();
}
