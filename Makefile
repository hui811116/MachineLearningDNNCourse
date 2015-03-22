obj=device_matrix.o cuda_memory_manager.o
CUDA_DIR=/usr/local/cuda/
LIBCUMATDIR=/tmp/libcumatrix/obj/
INCLUDE=-I /tmp/libcumatrix/include/\
	-I $(CUDA_DIR)include/\
	-I $(CUDA_DIR)samples/common/inc/
LD_LIBRARY=-L $(CUDA_DIR)/lib64
LIBRARY=-lcuda -lcublas -lcudart
TARGET=test.app

all: matMultTest.cpp
	g++ -o $(TARGET) matMultTest.cpp $(LIBCUMATDIR)device_matrix.o $(LIBCUMATDIR)cuda_memory_manager.o $(INCLUDE) $(LD_LIBRARY) $(LIBRARY)

clean:
	@rm -f $(TARGET) *o
