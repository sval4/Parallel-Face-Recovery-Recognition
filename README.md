# Parallel-Face-Recovery-Recognition
Parallel Implementation of Nearest Neighbor (image match) algorithm and linear interpolation (occlusion recovery) algorithm; using MPI and CUDA.
## How to Run (Ensure access to CUDA, MPI, and clockcycle.h):
- mpixlc -g face.c -c -o face-mpi.o
- nvcc -g -G face.cu -c -o face-cuda.o 
- mpicc -g face-mpi.o face-cuda.o -o face-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lm
- mpirun --bind-to core --report-bindings -np 1 face-exe 1 1 1024

The first argument chooses which training file for input (1 indicates training file with 1*360 images)

The second argument chooses which testing file for input (1 indicates training file with 1*360 images)

The third argument chooses number of threads per block for CUDA

How to run comments also present in face-serial.c, face-mpi.c, and face.c
## Test Cases:
### Sequential:
- Train 1, Test 1
- Train 2, Test 2
- Train 4, Test 4
- Train 8, Test 8
- Train 16, Test 16

### Strong Scaling Study 1 (Train 16, Test 16, No GPU, face-mpi.c):
- Rank 1 
- Ranks 2
- Ranks 4
- Ranks 8
- Ranks 12
- Ranks 24
- Ranks 36

### Strong Scaling Study 2 (Train 16, Test 16, 1 GPU):
- Ranks 1, Blocksize 1
- Ranks 2, Blocksize 8
- Ranks 4, Blocksize 16
- Ranks 8, Blocksize 32
- Ranks 12, Blocksize 128
- Ranks 24, Blocksize 512
- Ranks 36, Blocksize 1024

### Weak Scaling Study(Blocksize 1024):
- 1 GPU/Rank, Train 1, Test 1
- 2 GPU/Rank, Train 2, Test 2 
- 3 GPU/Rank, Train 3, Test 3 
- 4 GPU/Rank, Train 4, Test 4
- 5 GPU/Rank, Train 5, Test 5 
- 6 GPU/Rank, Train 6, Test 6