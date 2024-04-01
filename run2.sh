#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=vals@rpi.edu
#SBATCH --mail-type=ALL
module load xl_r spectrum-mpi cuda/11.2
gcc -g face-serial.c -lm
./a.out 16 16 1