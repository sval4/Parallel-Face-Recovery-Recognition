ssh PCPEvlsh@dcsfen01


make -f run.mk


salloc --qos=interactive -t 00:30:00 --gres=gpu:1 srun --pty bash -i

squeue

module load xl_r spectrum-mpi cuda/11.2

mpirun -np 4 ./highlife-mpi 5 8 5 0

sbatch --mail-type=ALL --mail-user=vals@rpi.edu ./script.sh




salloc -N 1 --partition=el8-rpi --gres=gpu:1 -t 60

make -f run.mk

module load xl_r spectrum-mpi cuda/11.2
mpirun --bind-to core --report-bindings -np 1 ./face-exe 16 16 128
mpirun -np 1 ./face-exe 1 1 32

# 35,298,625,909.000000
# 148,666,483,417.000000
