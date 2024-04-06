#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 00:30:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --mail-user=vals@rpi.edu
#SBATCH --mail-type=ALL
module load xl_r spectrum-mpi cuda/11.2
srun hostname -s | sort -u > hostfile
while IFS= read -r line
do
        echo "$line slots=18"
done < hostfile > temp && mv temp hostfile
make -f run.mk
mpirun -hostfile hostfile face-exe 16 16 1024