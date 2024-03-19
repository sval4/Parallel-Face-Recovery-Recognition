#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#define MAX_LINE_LENGTH 1000000
#define NUM_ROWS 360
#define NUM_COLS 4096
#define PATCH_SIZE 3 /* Patch size for neighborhood */

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
    // Iterate over each pixel in the image
    for(i = 0; i < NUM_COLS; i++){
        if(occlusion_mask[i] == 1){
            int left = i - 1;
            while(left >= 0 && left % NUM_COLS != (NUM_COLS - 1) && occlusion_mask[left] == 1) left--;
            int right = i + 1;
            while(right < NUM_COLS && right % NUM_COLS != 0 && occlusion_mask[right] == 1) right++;
            int top = i - NUM_COLS;
            while(top >= 0 && occlusion_mask[top] == 1) top-= NUM_COLS;
            int bottom = i + NUM_COLS;
            while(bottom < NUM_COLS && occlusion_mask[bottom] == 1) bottom+= NUM_COLS;

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
    return index;
}

int main() {
    srand(123);
    FILE *file;
    char line[MAX_LINE_LENGTH];
    char *token;
    double* train_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* test_images = calloc(40 * NUM_COLS, sizeof(double));
    double* org_images = calloc(NUM_ROWS * NUM_COLS, sizeof(double));
    double* means = calloc(NUM_COLS, sizeof(double));
    int* train_labels = calloc(NUM_ROWS, sizeof(int));
    int* test_labels = calloc(40, sizeof(int));
    int field_count;

    // Open the CSV file for reading
    file = fopen("faces_train.csv", "r");
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

    colMeans(train_images, means);


//----------------------------------------------------------------------------------------------
    // Open the CSV file for reading
    file = fopen("faces_test.csv", "r");
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
    int i;
    int j;
    for(j=0; j < NUM_COLS; j++){
        for(i = 0; i < 40; i++){
            test_images[i * NUM_COLS + j] = test_images[i * NUM_COLS + j] - means[j];
        }
    }
//----------------------------------------------------------------------------------------------
    int correctCount = 0;
    for(i = 0; i < 40; i++){
        double* ref_image = calloc(NUM_COLS, sizeof(double));
        for(j = i * NUM_COLS; j < i * NUM_COLS + NUM_COLS; j++){
            ref_image[j - (i * NUM_COLS)] = test_images[j];
        }
        int index = matchImage(train_images, ref_image);
        if(train_labels[index] == test_labels[i]){
            correctCount++;
        }
        free(ref_image);
    }

    printf("Success Rate: %lf%%\n", 100 * (double) correctCount/40);

    FILE* pyFile = fopen("output.txt", "w");
    if (pyFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    correctCount = 0;
    int index = 0;
    for(j = 0; j < 40; j++){
        int* occlusion_mask = calloc(NUM_COLS, sizeof(int));
        for(i =0; i < NUM_COLS; i++){
            occlusion_mask[i] = rand() % 2;
        }
        double* occluded_image = calloc(NUM_COLS, sizeof(double));
        for(i = 0; i < NUM_COLS; i++){
            occluded_image[i] = test_images[j * NUM_COLS + i];
        }
        recover_occluded(occluded_image, occlusion_mask);

        index = matchImage(train_images, occluded_image);
        if(train_labels[index] == test_labels[j]){
            correctCount++;
        }
        free(occlusion_mask);
        free(occluded_image);
    }


    printf("Success Rate Occlusion: %lf%%\n", 100 * (double) correctCount/40);

    // Write each double to the file
    for(j = 0; j < NUM_COLS; j++){
        fprintf(pyFile, "%lf\n", org_images[index * NUM_COLS + j]);
    }

    fclose(pyFile);

    system("python3 display.py");

    free(train_images);
    free(test_images);
    free(org_images);
    free(means);
    free(train_labels);
    free(test_labels);

    return 0;
}
