all: roi_align_kernel.cu roi_align_kernel.h
	nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

#clean:
#	rm roi_align.cu.o

# nvcc compiler flags
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options