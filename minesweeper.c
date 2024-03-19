#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 8
#define COLS 8
#define MINES 7

char infoBoard[ROWS * COLS];
char displayBoard[ROWS * COLS];
size_t numRevealed = ROWS * COLS - MINES;


// __global__ void initializeBoard() {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < ROWS * COLS) {
//         infoBoard[tid] = '0';
//         displayBoard[tid] = 'X';
//     }
//     if (tid == 0) {
//         numRevealed = ROWS * COLS - MINES;
//     }
// }

// __global__ void placeMines() {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < MINES) {
//         int index;
//         do {
//             index = rand() % (ROWS * COLS);
//         } while (infoBoard[index] == '*');
//         infoBoard[index] = '*';
//     }
//     __syncthreads(); Might not need this but will need to do cuda device synchornize after placeMines method is called

//Probably need this in a different function than what is at the top
//     // Update adjacent cell counts
//     if (tid < ROWS * COLS) {
//         int row = tid / COLS;
//         int col = tid % COLS;
//         if (infoBoard[tid] != '*') {
//             for (int i = -1; i <= 1; i++) {
//                 for (int j = -1; j <= 1; j++) {
//                     int new_row = row + i;
//                     int new_col = col + j;
//                     if (new_row >= 0 && new_row < ROWS && new_col >= 0 && new_col < COLS &&
//                         infoBoard[new_row * COLS + new_col] == '*') {
//                         infoBoard[tid]++;
//                     }
//                 }
//             }
//         }
//     }
// }

void initializeBoard() {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            infoBoard[i * ROWS + j] = '0';
            displayBoard[i * ROWS + j] = 'X';
        }
    }
}

void placeMines() {
    for (int i = 0; i < MINES; i++) {
        int row, col;
        do {
            row = rand() % ROWS;
            col = rand() % COLS;
        } while (infoBoard[row * ROWS + col] == '*');
        infoBoard[row * ROWS + col] = '*';
        if(row * ROWS + col - 1 >= 0){
            if(infoBoard[row * ROWS + col - 1] != '*'){infoBoard[row * ROWS + col - 1]++;}
        }
        if(row * ROWS + col + 1 < ROWS * COLS){
            if(infoBoard[row * ROWS + col + 1] != '*'){infoBoard[row * ROWS + col + 1]++;}
        } 

        if((row - 1) * ROWS + col >= 0){
            if(infoBoard[(row - 1) * ROWS + col] != '*'){infoBoard[(row - 1) * ROWS + col]++;}
            if(row * ROWS + col - 1 >= 0){
                if(infoBoard[(row - 1) * ROWS + col - 1] != '*'){infoBoard[(row - 1) * ROWS + col - 1]++;}
            }
            if(row * ROWS + col + 1 < ROWS * COLS){
                if(infoBoard[(row - 1) * ROWS + col + 1] != '*'){infoBoard[(row - 1) * ROWS + col + 1]++;}
            }
        }
        if((row + 1) * ROWS + col < ROWS * COLS){
            if(infoBoard[(row + 1) * ROWS + col] != '*'){infoBoard[(row + 1) * ROWS + col]++;}
            if(row * ROWS + col - 1 >= 0){
                if(infoBoard[(row + 1) * ROWS + col - 1] != '*'){infoBoard[(row + 1) * ROWS + col - 1]++;}
            }
            if(row * ROWS + col + 1 < ROWS * COLS){
                if(infoBoard[(row + 1) * ROWS + col + 1] != '*'){infoBoard[(row + 1) * ROWS + col + 1]++;}
            }

        }

    }
}

void revealCell(int row, int col) {
    if (row * ROWS + col < 0 || row * ROWS + col >= ROWS * COLS) return;
    if (displayBoard[row * ROWS + col] != 'X') return;
    if(infoBoard[row * ROWS + col] != '*'){
        displayBoard[row * ROWS + col] = infoBoard[row * ROWS + col];
        numRevealed--; //Need this to be atomic not displayBoard[row * ROWS + col] = infoBoard[row * ROWS + col];
    }
    if (infoBoard[row * ROWS + col] == '0') {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                revealCell(row + i, col + j);
            }
        }
    }
}

void printBoard() {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%c ", displayBoard[i * ROWS + j]);
        }
        printf("\n");
    }
}

void printInfoBoard() {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%c ", infoBoard[i * ROWS + j]);
        }
        printf("\n");
    }
}

//For simulation, to keeps thing consistent the user will enter the location of every non mine cell
//srand used for mine placement and keeping the boards equal
// 20% of cells should be mines: ROWS * COLS * 0.2

int main() {
    srand(123);
    initializeBoard();
    printBoard();
    placeMines();
    printf("\n");
    printInfoBoard();

    int gameOver = 0;
    int row, col;

    while(!gameOver){
        do{
            printf("Enter the row and column to click (0-7): ");
            scanf("%d %d", &row, &col);
        }while(row * ROWS + col < 0 || row * ROWS + col >= ROWS * COLS);

        if(infoBoard[row * ROWS + col] == '*'){
            displayBoard[row * ROWS + col] = '*';
            printBoard();
            printf("You hit a mine! Game over.\n");
            return 0;
        }
        revealCell(row, col);
        printBoard();
        printf("\n");

        if (numRevealed == 0) {
            printf("You Won!\n");
            return 0;
        }
    }

    return 0;
}
