#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <mpi.h>
#include "clockcycle.h"

#define MAX_LINE_LENGTH (4097 * 3 + 4096)
#define NUM_COLS 4096

double* occluded_image = NULL;
int* occluded_mask = NULL;
double* test_images = NULL;
double *train_images = NULL;
double *dist = NULL;
size_t threads_per_block = 0;
size_t NUM_ROWS = 0;
size_t TEST_NUM_ROWS = 0;

extern void occludedKernelLaunch(double** data, double* test_images, size_t rows, size_t cols, size_t threadsCount);
extern void createData(size_t length, size_t cols, size_t test_length);
extern void copyToCPUOccluded(double* data, size_t length, int start);
extern void createMask(int** data, size_t rows, size_t cols, size_t threadsCount);
extern void normalizeKernelLaunch(double** data, size_t rows, size_t cols, size_t threadsCount);
extern void meanCenterKernelLaunch(double** data, double* means, size_t rows, size_t cols, size_t threadsCount);
extern void norm2KernelLaunch(double** data, double* test, size_t rows, size_t cols, size_t start, size_t threadsCount, double** answer);

void colMeans(double* images, double* means, int rows_per_rank){
    int i, j;
    double local_sum[NUM_COLS] = {0};

    for (j = 0; j < NUM_COLS; j++) { //For each column
        for (i = 0; i < rows_per_rank; i++) { //Loop through each row for the rank and add up all pixels
            local_sum[j] += images[i * NUM_COLS + j];
        }
        MPI_Allreduce(MPI_IN_PLACE, &local_sum[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        means[j] = local_sum[j] / NUM_ROWS;
    }
}

double norm2(double* images, int start, double* refImage, int start_occ){
    int i;
    double sum = 0;
    for(i = start; i < NUM_COLS + start; i++){
        sum += (images[i] - refImage[start_occ + i - start]) * (images[i] - refImage[start_occ + i - start]);
    }
    return sqrt(sum);
}

//CUDA version, but it might be slower
// int matchImage(double* images, double* refImage, int rows_per_rank, int start){
//     int i;
//     double minDist = DBL_MAX;
//     int index = 0;
//     norm2KernelLaunch(&images, refImage, rows_per_rank, NUM_COLS, start, threads_per_block, &dist);
//     for(i = 0; i < rows_per_rank; i++){
//         if(minDist > dist[i]){
//             minDist = dist[i];
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


int matchImage(double* images, double* refImage, int rows_per_rank, int start){
    int i;
    double minDist = DBL_MAX;
    int index = 0;
    for(i = 0; i < rows_per_rank; i++){
        double distance = norm2(images, i*NUM_COLS, refImage, start);
        if(minDist > distance){
            minDist = distance;
            index = i;
        }
    }
    double global_minDist;
    MPI_Allreduce(&minDist, &global_minDist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    // if(myrank == 0){printf("GlobMinDist:%lf\n", global_minDist);}
    if (minDist > global_minDist) {
        index = -1;
    }
    return index;
}

int main(int argc, char *argv[]) {
    srand(123);
    int train_file_num = atoi(argv[1]);
    int test_file_num = atoi(argv[2]);
    threads_per_block = atoi(argv[3]);
    NUM_ROWS = 360 * train_file_num;
    TEST_NUM_ROWS = 360 * test_file_num;
    int myrank, numranks;
    MPI_File file;
    MPI_Status status;
    double overall_start_time = 0;
    double input_start_time = 0;
    double output_start_time = 0;
    uint64_t overall_start_cycle = 0;
    uint64_t input_start_cycle = 0;
    uint64_t output_start_cycle = 0;
    char line[MAX_LINE_LENGTH + 2];
    char *token;
    double* org_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* means = calloc(NUM_COLS, sizeof(double)); //Means of columns of train_images
    int* train_labels = calloc(NUM_ROWS, sizeof(int));
    int* test_labels = calloc(TEST_NUM_ROWS, sizeof(int));
    int field_count;
    int i,j;
    char train_file[1000];
    char test_file[1000];


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    sprintf(train_file, "faces_train360x%d.csv", train_file_num);
    sprintf(test_file, "faces_testx%d.csv", test_file_num);

    int rows_per_rank = NUM_ROWS / numranks;
    int start_row = myrank * rows_per_rank;

    createData(rows_per_rank, NUM_COLS, TEST_NUM_ROWS); //Cuda Malloc Managed call

    // +2 to handle the \r\n at end of each line
    MPI_Offset offset = start_row * (MAX_LINE_LENGTH + 2);

    if (myrank == 0) {
        overall_start_cycle = clock_now();
        input_start_cycle = overall_start_cycle;
        output_start_cycle = overall_start_cycle;

        overall_start_time = MPI_Wtime();
        input_start_time = overall_start_time;
        output_start_time = overall_start_time;
    }
    MPI_File_open(MPI_COMM_WORLD, train_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    if (file == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        return 1;
    }
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
            org_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            train_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }
        // if(myrank == 0){printf("Count: %d\n", field_count);}
    }


    MPI_File_close(&file);
    if (myrank == 0) {
        uint64_t t2 = clock_now();
        printf("Input Cycle for Train is: %lf cycles\n", t2 - input_start_cycle);
        double t3 = MPI_Wtime();
        printf("Input Time for Train is: %lf seconds\n", t3 - input_start_time);
    }
    normalizeKernelLaunch(&train_images, rows_per_rank, NUM_COLS, threads_per_block);
    colMeans(train_images, means, rows_per_rank);
    meanCenterKernelLaunch(&train_images, means, rows_per_rank, NUM_COLS, threads_per_block);
    if(myrank == 1){
        // for(i = 0; i < NUM_COLS; i++){
        //     printf("Mean[%d]: %lf\n",i, means[i]);
        // }
    }



// //----------------------------------------------------------------------------------------------
    //Number TEST_NUM_ROWS is always the number of images in test dataset
    int test_rows_per_rank = TEST_NUM_ROWS / numranks;
    int test_start_row = myrank * test_rows_per_rank;

    // +2 to handle the \r\n
    offset = test_start_row * (MAX_LINE_LENGTH + 2);
    if (myrank == 0) {
        input_start_cycle = clock_now();
        input_start_time = MPI_Wtime();
    }
    MPI_File_open(MPI_COMM_WORLD, test_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
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
            test_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }
        // if(myrank == 0){printf("Count: %d\n", field_count);}
    }
    MPI_File_close(&file);
    if (myrank == 0) {
        uint64_t t2 = clock_now();
        printf("Input Cycle for Test is: %lf cycles\n", t2 - input_start_cycle);
        double t3 = MPI_Wtime();
        printf("Input Time for Test is: %lf seconds\n", t3 - input_start_time);
    }
    normalizeKernelLaunch(&test_images, TEST_NUM_ROWS, NUM_COLS, threads_per_block);
    meanCenterKernelLaunch(&test_images, means, TEST_NUM_ROWS, NUM_COLS, threads_per_block);
    MPI_Allgather(test_images + (test_start_row * NUM_COLS), test_rows_per_rank * NUM_COLS, MPI_DOUBLE, test_images, test_rows_per_rank * NUM_COLS, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(test_labels + test_start_row, test_rows_per_rank, MPI_INT, test_labels, test_rows_per_rank, MPI_INT, MPI_COMM_WORLD);

//----------------------------------------------------------------------------------------------
    if (myrank == 0) {
        output_start_cycle = clock_now();
        output_start_time = MPI_Wtime();
    }
    MPI_File_open(MPI_COMM_WORLD, "match.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    int correctCount = 0;
    for(i = 0; i < TEST_NUM_ROWS; i++){
        double* ref_image = calloc(NUM_COLS, sizeof(double));
        for(j = i * NUM_COLS; j < i * NUM_COLS + NUM_COLS; j++){ //Ref image is a given test image
            ref_image[j - (i * NUM_COLS)] = test_images[j];
        }
        int index = matchImage(train_images, ref_image, rows_per_rank, 0); // Match ref image with closest training image

        char outputString[150];

        if(index != -1 && train_labels[index] == test_labels[i]){
            correctCount++;
            sprintf(outputString, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct", myrank * rows_per_rank + index, i, train_labels[index]);
        }else{
            if(index == -1){
                sprintf(outputString, "Train-Image: %d, Test-Image: %d, Found Better", index, i);
            }else{
                sprintf(outputString, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct-Label: %d, Wrong", myrank * rows_per_rank + index, i, train_labels[index], test_labels[i]);
            }
        }
        offset = myrank * TEST_NUM_ROWS * 150 + i * 150;// Assuming each rank writes 150 characters
        if (strlen(outputString) < 150) {
            sprintf(outputString + strlen(outputString), "%*s", 150 - (int) strlen(outputString), " ");
            outputString[strlen(outputString) - 2] = '\n';
            outputString[strlen(outputString) - 1] = '\0';
        }
        MPI_File_write_at(file, offset, outputString, strlen(outputString)+ 1, MPI_CHAR, MPI_STATUS_IGNORE);
        free(ref_image);
    }
    int global_correctCount;
    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate: %lf%%\n", (100 * (double) global_correctCount/TEST_NUM_ROWS)/train_file_num);}
    MPI_File_close(&file);
    if (myrank == 0) {
        uint64_t t2 = clock_now();
        printf("Output Cycle for Match is: %lf cycle\n", t2 - output_start_cycle);
        double t3 = MPI_Wtime();
        printf("Output Time for Match is: %lf seconds\n", t3 - output_start_time);
    }

    correctCount = 0;
    int index = 0;
    if (myrank == 0) {
        output_start_cycle = clock_now();
        output_start_time = MPI_Wtime();
    }
    MPI_File_open(MPI_COMM_WORLD, "occlusion_recovery.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    for(j = 0; j < TEST_NUM_ROWS; j++){
        if(myrank == 0 && j == 0){
            createMask(&occluded_mask, TEST_NUM_ROWS, NUM_COLS, threads_per_block);
            occludedKernelLaunch(&occluded_image, test_images, TEST_NUM_ROWS, NUM_COLS, threads_per_block);
            // copyToCPUOccluded(occluded_image_local, TEST_NUM_ROWS * NUM_COLS, 0); Might not need this
        }
        if(j == 0){
            MPI_Bcast(occluded_image, TEST_NUM_ROWS * NUM_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        index = matchImage(train_images, occluded_image, rows_per_rank, j*NUM_COLS);

        char outputString[150];

        if(index != -1 && train_labels[index] == test_labels[j]){
            correctCount++;
            sprintf(outputString, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct", myrank * rows_per_rank + index, j, train_labels[index]);
        }else{
            if(index == -1){
                sprintf(outputString, "Train-Image: %d, Test-Image: %d, Found Better", index, j);
            }else{
                sprintf(outputString, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct-Label: %d, Wrong", myrank * rows_per_rank + index, j, train_labels[index], test_labels[j]);
            }
        }
        offset = myrank * TEST_NUM_ROWS * 150 + j * 150;// Assuming each rank writes 150 characters
        if (strlen(outputString) < 150) {
            sprintf(outputString + strlen(outputString), "%*s", 150 - (int) strlen(outputString), " ");
            outputString[strlen(outputString) - 2] = '\n';
            outputString[strlen(outputString) - 1] = '\0';
        }
        MPI_File_write_at(file, offset, outputString, strlen(outputString)+ 1, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
    if (myrank == 0) {
        uint64_t t2 = clock_now();
        printf("Output Cycle for Occlusion_Recovery is: %lf cycle\n", t2 - output_start_cycle);
        double t3 = MPI_Wtime();
        printf("Output Time for Occlusion_Recovery is: %lf seconds\n", t3 - output_start_time);
    }

    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate Occlusion: %lf%%\n", (100 * (double) global_correctCount/TEST_NUM_ROWS)/train_file_num);}

    if (myrank == 0) {
        uint64_t t2 = clock_now();
        printf("Overall Cycle is: %lf cycle\n", t2 - overall_start_cycle);
        double t3 = MPI_Wtime();
        printf("Overall Time is: %lf seconds\n", t3 - output_start_time);
    }

    // if(myrank == 0){system("python3 display.py");}

    // free(train_images);
    // free(test_images);
    free(org_images);
    free(means);
    free(train_labels);
    free(test_labels);

    MPI_Finalize();
    return 0;
}
