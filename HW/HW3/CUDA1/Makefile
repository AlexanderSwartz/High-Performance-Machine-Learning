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
OPTIONS := -O3 -std=c++17

.PHONY: all part-a part-b part-c clean

all: part-a part-b part-c

########################################################################
# Part-A: build timer.o, vecadd variants, matmult variants
########################################################################


### Kernel for adding two vectors without using coalesced memory
### Each thread adds ValuesPerThread elements, determined by user input in vecadd.cu
#######################################################################
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

# matmult variants (if matmult.cu and kernels exist in this dir)
#######################################################################
## Provided Kernel
MAT_K00 := matmultKernel00.o
MAT_K01 := matmultKernel01.o

$(MAT_K00) : matmultKernel00.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu  matmultKernel.h $(MAT_K00) timer.o
	${NVCC} $< $(MAT_K00) -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
$(MAT_K01) : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h $(MAT_K01) timer.o
	${NVCC} $< $(MAT_K01) -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32

timer.o: timer.cu timer.h
	$(NVCC) $< -c -o $@ $(OPTIONS)

part-a: vecadd00 vecadd01 matmult00 matmult01

########################################################################
# Part-B (compile files placed in ../Part-B) - build only if sources exist
########################################################################
PB_DIR := ../Part-B
part-b: ../Part-B/q1 ../Part-B/q2 ../Part-B/q3

#######################################################################
../Part-B/q1: ../Part-B/q1.cpp timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

../Part-B/q2: ../Part-B/q2.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

../Part-B/q3: ../Part-B/q3.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS) -lm

########################################################################
# Part-C (compile files placed in ../Part-C)
########################################################################
PC_DIR := ../Part-C
part-c: ../Part-C/c1 ../Part-C/c2 ../Part-C/c3

#######################################################################
../Part-C/c1: ../Part-C/c1.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS)

../Part-C/c2: ../Part-C/c2.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS)

../Part-C/c3: ../Part-C/c3.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS) -lcudnn

########################################################################
clean:
	rm -f vecadd00 vecadd01 matmult00 matmult01 vecaddKernel00.o vecaddKernel01.o $(MAT_K00) $(MAT_K01) timer.o
	@echo "Cleaned Part-A artifacts"
