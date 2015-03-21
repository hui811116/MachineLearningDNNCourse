obj=device_matrix.o cuda_memory_manager.o
CUDA_DIR=/usr/local/cuda/
LIBCUMATDIR=~/libcumatrix-master/obj/
INCLUDE=-I ~/libcumatrix-master/include/\
	-I $(CUDA_DIR)include/\
	-I $(CUDA_DIR)samples/common/inc/
LD_LIBRARY=-L $(CUDA_DIR)/lib64
LIBRARY=-lcuda -lcublas -lcudart

test: matMultTest.cpp
	g++ -o test matMultTest.cpp $(LIBCUMATDIR)device_matrix.o $(LIBCUMATDIR)cuda_memory_manager.o $(INCLUDE) $(LD_LIBRARY) $(LIBRARY)

clean:
	@rm -f test
