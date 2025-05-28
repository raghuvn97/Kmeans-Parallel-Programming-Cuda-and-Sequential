# Makefile for CUDA K-Means Clustering with CUDA 11.5
NVCC = nvcc

# Compiler flags optimized for CUDA 11.5
NVCCFLAGS = -arch=sm_80 -O2
LDFLAGS = -lcudart
TARGET = kmeans_cuda

all: $(TARGET)

$(TARGET): kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)