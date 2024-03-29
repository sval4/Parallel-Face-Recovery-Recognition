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

// Function to recover occluded regions using linear interpolation
void recover_occluded(double* occluded_image, int* occlusion_mask) {
    int i;
    int width = sqrt(NUM_COLS);
    // Iterate over each pixel in the image
    for(i = 0; i < NUM_COLS; i++){
        if(occlusion_mask[i] == 1){
            int left = i - 1;
            while(left >= 0 && left % width != (width - 1) && occlusion_mask[left] == 1) left--;
            int right = i + 1;
            while(right < NUM_COLS && right % width != 0 && occlusion_mask[right] == 1) right++;
            int top = i - width;
            while(top >= 0 && occlusion_mask[top] == 1) top-= width;
            int bottom = i + width;
            while(bottom < NUM_COLS && occlusion_mask[bottom] == 1) bottom+= width;

            // Perform linear interpolation
            double interpolated_value = 0.0;
            int count = 0;
            if(left >= 0 && occlusion_mask[left] == 0){
                interpolated_value += occluded_image[left];
                count++;
            }
            if(right < NUM_COLS && occlusion_mask[right] == 0){
                interpolated_value += occluded_image[right];
                count++;
            }
            if(top >= 0 && occlusion_mask[top] == 0){
                interpolated_value += occluded_image[top];
                count++;
            }
            if(bottom < NUM_COLS && occlusion_mask[bottom] == 0){
                interpolated_value += occluded_image[bottom];
                count++;
            }
            if(count > 0){
                occluded_image[i] = interpolated_value / count;
            }
        }
    }
}

void normalize(double* row, int start){
    int i;
    double sum = 0;
    for(i = start; i < NUM_COLS + start; i++){
        sum += row[i] * row[i];
    }

    for(i=start; i < NUM_COLS + start; i++){
        row[i] = row[i]/sqrt(sum);
    }
    
}

void colMeans(double* images, double* means, int rows_per_rank){
    int i, j;
    double local_sum[NUM_COLS] = {0};

    for (j = 0; j < NUM_COLS; j++) {
        for (i = 0; i < rows_per_rank; i++) {
            local_sum[j] += images[i * NUM_COLS + j];
        }
        MPI_Allreduce(MPI_IN_PLACE, &local_sum[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        means[j] = local_sum[j] / NUM_ROWS;
        for (i = 0; i < rows_per_rank; i++) {
            images[i * NUM_COLS + j] -= means[j];
        }
    }
}

double norm2(double* images, int start, double* refImage){
    int i;
    double sum = 0;
    for(i = start; i < NUM_COLS + start; i++){
        sum += (images[i] - refImage[i - start]) * (images[i] - refImage[i - start]);
    }
    return sqrt(sum);

}

int matchImage(double* images, double* refImage, int rows_per_rank){
    int i;
    double minDist = DBL_MAX;
    int index = 0;
    for(i = 0; i < rows_per_rank; i++){
        double dist = norm2(images, i * NUM_COLS, refImage);
        if(minDist > dist){
            minDist = dist;
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
    int myrank, numranks;
    MPI_File file;
    MPI_Status status;
    double start_time = 0;
    char line[MAX_LINE_LENGTH + 2];
    char *token;
    double* train_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* test_images = calloc(40 * NUM_COLS, sizeof(double));
    double* org_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* means = calloc(NUM_COLS, sizeof(double));
    int* train_labels = calloc(NUM_ROWS, sizeof(int));
    int* test_labels = calloc(40, sizeof(int));
    int field_count;
    int i,j;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

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
        normalize(train_images, i * NUM_COLS);
    }


    MPI_File_close(&file);

    colMeans(train_images, means, rows_per_rank);
    if(myrank == 1){
        // for(i = 0; i < NUM_COLS; i++){
        //     printf("Mean[%d]: %lf\n",i, means[i]);
        // }
    }



// //----------------------------------------------------------------------------------------------

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
            test_images[i * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }
        // if(myrank == 0){printf("Count: %d\n", field_count);}
        normalize(test_images, i * NUM_COLS);
    }
    MPI_File_close(&file);
    for(j=0; j < NUM_COLS; j++){
        for(i = test_start_row; i < test_start_row + test_rows_per_rank; i++){
            test_images[i * NUM_COLS + j] = test_images[i * NUM_COLS + j] - means[j];
        }
    }
    MPI_Allgather(test_images + (test_start_row * NUM_COLS), test_rows_per_rank * NUM_COLS, MPI_DOUBLE, test_images, test_rows_per_rank * NUM_COLS, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(test_labels + test_start_row, test_rows_per_rank, MPI_INT, test_labels, test_rows_per_rank, MPI_INT, MPI_COMM_WORLD);

//----------------------------------------------------------------------------------------------
    MPI_File_open(MPI_COMM_WORLD, "match.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    
    
    int correctCount = 0;
    for(i = 0; i < 40; i++){
        double* ref_image = calloc(NUM_COLS, sizeof(double));
        for(j = i * NUM_COLS; j < i * NUM_COLS + NUM_COLS; j++){
            ref_image[j - (i * NUM_COLS)] = test_images[j];
        }
        
        int index = matchImage(train_images, ref_image, rows_per_rank);

        char outputString[200];
        if(index != -1 && train_labels[index] == test_labels[i]){
            correctCount++;
            sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Predicted-Label: %d, Correct\n", myrank, i, index, train_labels[index]);
        }else{
            if(index == -1){
                sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Found Better\n", myrank, i, index);
            }else{
                sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Predicted-Label: %d, Correct-Label: %d, Wrong\n", myrank, i, index, train_labels[index], test_labels[i]);
            }
        }
        offset = myrank * 40 * 200 + i * 200;// Assuming each rank writes 100 characters
        MPI_File_write_at(file, offset, outputString, strlen(outputString), MPI_CHAR, MPI_STATUS_IGNORE);
        free(ref_image);
    }
    int global_correctCount;
    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate: %lf%%\n", 100 * (double) global_correctCount/40);}
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);

    FILE* pyFile = fopen("output.txt", "w");
    if (pyFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    correctCount = 0;
    int index = 0;
    MPI_File_open(MPI_COMM_WORLD, "occlusion_recovery.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    for(j = 0; j < 40; j++){
        int* occlusion_mask = calloc(NUM_COLS, sizeof(int));
        double* occluded_image = calloc(NUM_COLS, sizeof(double));
        if(myrank == 0){
            for(i = 0; i < NUM_COLS; i++){
                occlusion_mask[i] = rand() % 2;
            }
            for(i = 0; i < NUM_COLS; i++){
                occluded_image[i] = test_images[j * NUM_COLS + i];
            }
            recover_occluded(occluded_image, occlusion_mask);
        }
        MPI_Bcast(occluded_image, NUM_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        index = matchImage(train_images, occluded_image, rows_per_rank);
        char outputString[200];
        if(index != -1 && train_labels[index] == test_labels[j]){
            correctCount++;
            sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Predicted-Label: %d, Correct\n", myrank, j, index, train_labels[index]);
        }else{
            if(index == -1){
                sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Found Better\n", myrank, j, index);
            }else{
                sprintf(outputString, "Rank: %d, Image: %d, Predicted-Index: %d, Predicted-Label: %d, Correct-Label: %d, Wrong\n", myrank, j, index, train_labels[index], test_labels[j]);
            }
        }
        offset = myrank * 40 * 200 + j * 200;// Assuming each rank writes 200 characters
        MPI_File_write_at(file, offset, outputString, strlen(outputString), MPI_CHAR, MPI_STATUS_IGNORE);
        free(occlusion_mask);
        free(occluded_image);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);


    MPI_Reduce(&correctCount, &global_correctCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0){printf("Success Rate Occlusion: %lf%%\n", 100 * (double) global_correctCount/40);}

    if (myrank == 0) {
        double t2 = MPI_Wtime();
        printf("Time is: %lf seconds\n", t2 - start_time);
    }

    if(myrank == 0){
        // Write each double to the file
        for(j = 0; j < NUM_COLS; j++){
            fprintf(pyFile, "%lf\n", org_images[NUM_COLS + j]);
        }
    }


    fclose(pyFile);

    // if(myrank == 0){system("python3 display.py");}

    free(train_images);
    free(test_images);
    free(org_images);
    free(means);
    free(train_labels);
    free(test_labels);

    MPI_Finalize();
    return 0;
}
