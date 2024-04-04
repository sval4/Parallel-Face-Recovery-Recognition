all: face.c face.cu clockcycle.h
	mpixlc -g face.c -c -o face-mpi.o
	nvcc -g -G face.cu -c -o face-cuda.o 
	mpicc -g face-mpi.o face-cuda.o -o face-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lm