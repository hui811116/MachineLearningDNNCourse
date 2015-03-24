CC=gcc
CXX=g++
CFLAGS=
NVCC=nvcc -arch=sm_21 -w

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=test.app
LIBCUMATDIR=/home/larry/Documents/MLDS/libcumatrix/
OBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/dataset.o obj/dnn.o
.PHONY: debug all clean o3
all: libs

o3: CFLAGS+=-o3
o3: all
debug: CFLAGS+=-g -DDEBUG

libs: $(OBJ) $(LIBCUMATDIR)lib/libcumatrix.a

$(LIBCUMATDIR)lib/libcumatrix.a: $(OBJ)
	rm -f $@
	ar rcs $@ $^
	ranlib $@

vpath %.h include/
vpath %.cpp src/

INCLUDE= -I include\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L $(CUDA_DIR)lib64
LIBRARY=-lcuda -lcublas -lcudart
TARGET=test.app

all: $(OBJ) $(HEADEROBJ) matMultTest.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(OBJ) $(LD_LIBRARY) $(LIBRARY)

debug: $(OBJ) $(HEADEROBJ) temp.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(OBJ)\
 $(LIBRARY) $(LD_LIBRARY) 

clean:
	@rm -f $(EXECUTABLES) obj/*

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $^
