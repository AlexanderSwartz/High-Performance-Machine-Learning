###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS := -O3 -std=c++17
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := as7629CUDA1.tar
EXECS := vecadd00 vecadd01 matmult00 matmult01 \
		 ../Part-B/q1 ../Part-B/q2 ../Part-B/q3 \
		 ../Part-C/c1 ../Part-C/c2 ../Part-C/c3
all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

########################################################################
# Part-A
########################################################################

# utility for measuring time before and after kernel calls
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

# Kernel for adding two vectors WITHOUT using coaelsced memory
# Each thread adds ValuesPerThread elements, determined by user input in vecadd.cu
#######################################################################
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# Executable for vecadd with non-coalesced memory access
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Kernel for adding two vectors WITH using coaelsced memory
# Accesses global memory with stride so that threads in the same warp access consecutive memory locations
# Each thread adds ValuesPerThread elements, determined by user input in vecadd.cu
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## Provided Kernel
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
## Uses thread coarsening so each thread computes 4 elements of output instead of 1
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32

########################################################################
# Part-B
########################################################################

# Implements vector addition on CPU as baseline
../Part-B/q1: ../Part-B/q1.cpp timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

# Implements vector addition on GPU under 3 different grid and block configurations
../Part-B/q2: ../Part-B/q2.cu timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

# Same as q2 but uses unified memory
../Part-B/q3: ../Part-B/q3.cu timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

########################################################################
# Part-C
########################################################################
# Implements convolution on GPU using a custom kernel
../Part-C/c1: ../Part-C/c1.cu timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS)

# Implements convolution on GPU using a custom kernel that uses tiling and shared memory
../Part-C/c2: ../Part-C/c2.cu timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS)

# Implements convolution on GPU using cuDNN library
../Part-C/c3: ../Part-C/c3.cu timer.o
	${NVCC} -I../Part-A $< timer.o -o $@ $(LIB) $(OPTIONS) -lcudnn
