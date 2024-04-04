#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=vals@rpi.edu
#SBATCH --mail-type=ALL
module load xl_r spectrum-mpi cuda/11.2
make -f run.mk
mpirun --bind-to core --report-bindings -np 1 face-exe 16 16 1