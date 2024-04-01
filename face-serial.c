#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <time.h>
#include "clockcycle.h"

#define MAX_LINE_LENGTH 1000000
#define NUM_COLS 4096
#define PATCH_SIZE 3 /* Patch size for neighborhood */

size_t NUM_ROWS = 0;
size_t TEST_NUM_ROWS = 0;
size_t threads_per_block = 0;

// // Function to perform image inpainting using Markov Random Fields
// void recover_occluded_image(double* image, double* mask, double* recovered_image) {
//     // Iterate over each pixel in the image
//     for (int y = 0; y < HEIGHT; y++) {
//         for (int x = 0; x < WIDTH; x++) {
//             int index = y * WIDTH + x;

//             // If the pixel is not occluded, copy the pixel value to the recovered image
//             if (mask[index] == 0) {
//                 recovered_image[index] = image[index];
//             } else {
//                 // If the pixel is occluded, estimate its value based on neighboring pixels

//                 // Initialize variables for estimating the pixel value
//                 double sum = 0.0;
//                 double weight_sum = 0.0;

//                 // Iterate over the patch centered at the current pixel
//                 for (int dy = -PATCH_SIZE / 2; dy <= PATCH_SIZE / 2; dy++) {
//                     for (int dx = -PATCH_SIZE / 2; dx <= PATCH_SIZE / 2; dx++) {
//                         // Calculate the coordinates of the neighboring pixel
//                         int nx = x + dx;
//                         int ny = y + dy;

//                         // Check if the neighboring pixel is within the image bounds
//                         if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT && mask[ny * WIDTH + nx] == 0) {
//                             // Calculate the weight based on the distance between pixels
//                             double distance = sqrt(dx * dx + dy * dy);
//                             double weight = exp(-distance); // Adjust the weight function as needed

//                             // Accumulate the weighted pixel value
//                             sum += weight * image[ny * WIDTH + nx];
//                             weight_sum += weight;
//                         }
//                     }
//                 }

//                 // Calculate the estimated pixel value as the weighted average of neighboring pixel values
//                 if (weight_sum > 0) {
//                     recovered_image[index] = sum / weight_sum;
//                 } else {
//                     // If no valid neighboring pixels are available, use a default value
//                     recovered_image[index] = 0.0; // You can adjust this default value as needed
//                 }
//             }
//         }
//     }
// }

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

void colMeans(double* images, double* means){
    int i,j;
    for(j = 0; j < NUM_COLS; j++){
        double avg = 0;
        for(i = 0; i < NUM_ROWS; i++){
            avg += images[i * NUM_COLS + j] / NUM_ROWS;
        }
        means[j] = avg;
        for(i = 0; i < NUM_ROWS; i++){
            images[i * NUM_COLS + j] =  images[i * NUM_COLS + j] - avg;
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

int matchImage(double* images, double* refImage){
    int i;
    double minDist = DBL_MAX;
    int index = 0;
    for(i = 0; i < NUM_ROWS; i++){
        double dist = norm2(images, i * NUM_COLS, refImage);
        if(minDist > dist){
            minDist = dist;
            index = i;
        }
    }
    // printf("GlobMinDist:%lf\n", minDist);    
    return index;
}

int main(int argc, char *argv[]) {
    srand(123);
    int train_file_num = atoi(argv[1]);
    int test_file_num = atoi(argv[2]);
    threads_per_block = atoi(argv[3]);
    NUM_ROWS = 360 * train_file_num;
    TEST_NUM_ROWS = 360 * test_file_num;
    FILE *file;
    char line[MAX_LINE_LENGTH];
    char *token;
    double* train_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* test_images = calloc(TEST_NUM_ROWS * NUM_COLS, sizeof(double));
    double* org_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* means = calloc(NUM_COLS, sizeof(double));
    int* train_labels = calloc(NUM_ROWS, sizeof(int));
    int* test_labels = calloc(TEST_NUM_ROWS, sizeof(int));
    int field_count;
    char train_file[1000];
    char test_file[1000];
    clock_t overall_start_time = 0;
    clock_t input_start_time = 0;
    clock_t output_start_time = 0;
    uint64_t overall_start_cycle = 0;
    uint64_t input_start_cycle = 0;
    uint64_t output_start_cycle = 0;

    sprintf(train_file, "faces_train360x%d.csv", train_file_num);
    sprintf(test_file, "faces_testx%d.csv", test_file_num);
    overall_start_cycle = clock_now();
    input_start_cycle = overall_start_cycle;
    output_start_cycle = overall_start_cycle;

    overall_start_time = clock();
    input_start_time = overall_start_time;
    output_start_time = overall_start_time;
    // Open the CSV file for reading
    file = fopen(train_file, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
//----------------------------------------------------------------------------------------------
    // Read each line of the file
    int image_num = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        // Remove newline character if present
        if (line[strlen(line) - 1] == '\n') {
            line[strlen(line) - 1] = '\0';
        }

        // Split the line into fields based on comma delimiter
        token = strtok(line, ",");
        field_count = 0;
        train_labels[image_num] = atoi(token);
        token = strtok(NULL, ",");
        while (token != NULL) {
            org_images[image_num * NUM_COLS + field_count] = strtod(token, NULL);
            train_images[image_num * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }

        normalize(train_images, image_num * NUM_COLS);

        image_num++;
    }
    // Close the file
    fclose(file);
    uint64_t t2 = clock_now();
    printf("Input Cycle for Train is: %ld cycles\n", t2 - input_start_cycle);
    clock_t t3 = clock();
    printf("Input Time for Train is: %lf seconds\n", (double) (t3 - input_start_time)/CLOCKS_PER_SEC);

    colMeans(train_images, means);
    // for(int i = 0; i < NUM_COLS; i++){
    //     printf("Mean[%d]: %lf\n",i, means[i]);
    // }



//----------------------------------------------------------------------------------------------
    // Open the CSV file for reading
    input_start_cycle = clock_now();
    input_start_time = clock();
    file = fopen(test_file, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Read each line of the file
    image_num = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        // Remove newline character if present
        if (line[strlen(line) - 1] == '\n') {
            line[strlen(line) - 1] = '\0';
        }

        // Split the line into fields based on comma delimiter
        token = strtok(line, ",");
        field_count = 0;
        test_labels[image_num] = atoi(token);
        token = strtok(NULL, ",");
        while (token != NULL) {
            test_images[image_num * NUM_COLS + field_count] = strtod(token, NULL);
            token = strtok(NULL, ",");
            field_count++;
        }

        normalize(test_images, image_num * NUM_COLS);

        image_num++;
    }
    // Close the file
    fclose(file);
    t2 = clock_now();
    printf("Input Cycle for Test is: %ld cycles\n", t2 - input_start_cycle);
    t3 = clock();
    printf("Input Time for Test is: %lf seconds\n", (double) (t3 - input_start_time)/CLOCKS_PER_SEC);
    int i;
    int j;
    for(j=0; j < NUM_COLS; j++){
        for(i = 0; i < TEST_NUM_ROWS; i++){
            test_images[i * NUM_COLS + j] = test_images[i * NUM_COLS + j] - means[j];
        }
    }
//----------------------------------------------------------------------------------------------
    int correctCount = 0;
    output_start_cycle = clock_now();
    output_start_time = clock();
    file = fopen("match.txt", "w");
    for(i = 0; i < TEST_NUM_ROWS; i++){
        double* ref_image = calloc(NUM_COLS, sizeof(double));
        for(j = i * NUM_COLS; j < i * NUM_COLS + NUM_COLS; j++){
            ref_image[j - (i * NUM_COLS)] = test_images[j];
        }
        int index = matchImage(train_images, ref_image);
        
        if(train_labels[index] == test_labels[i]){
            correctCount++;
            fprintf(file, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct\n", index, i, train_labels[index]);
        }else{
            fprintf(file, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct-Label: %d, Wrong\n", index, i, train_labels[index], test_labels[i]);
        }
        free(ref_image);
    }
    fclose(file);
    t2 = clock_now();
    printf("Output Cycle for Match is: %ld cycle\n", t2 - output_start_cycle);
    t3 = clock();
    printf("Output Time for Match is: %lf seconds\n", (double) (t3 - output_start_time)/CLOCKS_PER_SEC);
    printf("Success Rate: %lf%%\n", 100 * (double) correctCount/TEST_NUM_ROWS);

    FILE* pyFile = fopen("output.txt", "w");
    if (pyFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    correctCount = 0;
    file = fopen("occlusion_recovery.txt", "w");
    int index = 0;
    double* occluded_image = calloc(NUM_COLS, sizeof(double));
    for(j = 0; j < TEST_NUM_ROWS; j++){
        int* occlusion_mask = calloc(NUM_COLS, sizeof(int));
        for(i =0; i < NUM_COLS; i++){
            occlusion_mask[i] = rand() % 2;
        }
        for(i = 0; i < NUM_COLS; i++){
            occluded_image[i] = test_images[j * NUM_COLS + i];
        }
        recover_occluded(occluded_image, occlusion_mask);

        index = matchImage(train_images, occluded_image);
        if(train_labels[index] == test_labels[j]){
            correctCount++;
            fprintf(file, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct\n", index, j, train_labels[index]);
        }else{
            fprintf(file, "Train-Image: %d, Test-Image: %d, Predicted-Label: %d, Correct-Label: %d, Wrong\n", index, j, train_labels[index], test_labels[j]);
        }
        free(occlusion_mask);
    }
    fclose(file);
    t2 = clock_now();
    printf("Output Cycle for Occlusion_Recovery is: %ld cycle\n", t2 - output_start_cycle);
    t3 = clock();
    printf("Output Time for Occlusion_Recovery is: %lf seconds\n", (double) (t3 - output_start_time)/CLOCKS_PER_SEC);
    printf("Success Rate Occlusion: %lf%%\n", 100 * (double) correctCount/TEST_NUM_ROWS);

    t2 = clock_now();
    printf("Overall Cycle is: %ld cycle\n", t2 - overall_start_cycle);
    t3 = clock();
    printf("Overall Time is: %lf seconds\n", (double) (t3 - overall_start_time)/CLOCKS_PER_SEC);

    // Write each double to the file
    for(j = 0; j < NUM_COLS; j++){
        fprintf(pyFile, "%lf\n", occluded_image[j]);
    }

    fclose(pyFile);

    // system("python3 display.py");

    free(train_images);
    free(test_images);
    free(org_images);
    free(means);
    free(train_labels);
    free(test_labels);
    free(occluded_image);

    return 0;
}
