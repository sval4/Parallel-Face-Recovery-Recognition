# Parallel-Face-Recovery-Recognition
Parallel Implementation of Nearest Neighbor (image match) algorithm and linear interpolation (occlusion recovery) algorithm; using MPI and CUDA.
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

### Strong Scaling Study 2 (Train 16, Test 16, Rank 1, 1 GPU):
- Blocksize 1
- Blocksize 8
- Blocksize 16
- Blocksize 32
- Blocksize 128
- Blocksize 512
- Blocksize 1024

### Strong Scaling Study 3 (Train 16, Test 16, 1 GPU):
- Ranks 1, Blocksize 1
- Ranks 2, Blocksize 8
- Ranks 4, Blocksize 16
- Ranks 8, Blocksize 32
- Ranks 12, Blocksize 128
- Ranks 24, Blocksize 512
- Ranks 36, Blocksize 1024

### Weak Scaling Study(Blocksize 256):
- 1 GPU/Rank, Train 1, Test 1
- 2 GPU/Rank, Train 2, Test 2 
- 3 GPU/Rank, Train 3, Test 3 (Needs to be created)
- 4 GPU/Rank, Train 4, Test 4
- 5 GPU/Rank, Train 5, Test 5 (Needs to be created)
- 6 GPU/Rank, Train 6, Test 6 (Needs to be created)