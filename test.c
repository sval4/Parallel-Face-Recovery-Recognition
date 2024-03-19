#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

#define N 4

// Function to print the solved Sudoku grid
void printGrid(int grid[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            printf("%2d", grid[row][col]);
        printf("\n");
    }
    printf("----------------------------------\n");
}

// Function to check if the given digit is safe to place at the given position
bool isSafe(int grid[N][N], int row, int col, int num) {
    // Check if the digit is already present in the current row
    int x;
    for (x = 0; x < N; x++)
        if (grid[row][x] == num){return false;}
    
    // Check if the digit is already present in the current column
    for (int y = 0; y < N; y++)
        if (grid[y][col] == num){return false;}

    
    // Check if the digit is already present in the current 3x3 subgrid
    int startRow = row - (row % 2);
    int startCol = col - (col % 2);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (grid[i + startRow][j + startCol] == num){return false;}
    
    return true;
}

// Function to solve the Sudoku puzzle using backtracking
bool solveSudoku(int grid[N][N], int row) {
    // If we have reached the end of the grid, the Sudoku puzzle is solved
    if (row == N - 1 && col == N)
        return true;

    // Move to the next row if we have reached the end of the current row
    if (col == N) {
        row++;
        col = 0;
    }

    // If the current cell is already filled, move to the next cell
    if (grid[row][col] != 0)
        return solveSudoku(grid, row, col + 1);

    // Try placing digits from 1 to 9 in the current cell
    for (int num = 1; num <= N; num++) {
        // Check if it's safe to place the digit at the current position
        if (isSafe(grid, row, col, num)) {
            // If it's safe, assign the digit to the current cell
            grid[row][col] = num;

            // Recursively solve the remaining puzzle
            if (solveSudoku(grid, row, col + 1))
                return true;

            // If the solution fails, backtrack and try the next digit
            grid[row][col] = 0;
        }
    }

    // If no digit can be placed at the current position, return false
    return false;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0)
            printf("Number of MPI processes must be equal to %d\n", N);
        MPI_Finalize();
        return 1;
    }

    // int grid[N][N] = {
    //     {0, 0, 0, 2},
    //     {0, 0, 0, 0},
    //     {3, 0, 0, 0},
    //     {0, 4, 0, 0}
    // };
    int grid[N][N] = {0};

    if (solveSudoku(grid, 0)) {
        if (rank == 0) {
            printf("Sudoku puzzle solved:\n");
            printGrid(grid);
        }
    } else {
        if (rank == 0)
            printf("No solution exists!\n");
    }

    MPI_Finalize();
    return 0;
}
