CC=gcc
CXX=g++
CFLAGS=

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=
LIBCUMATDIR=/tmp/libcumatrix/
OBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o

.PHONY: debug all clean o3
all: libs

o3: CFLAGS+=-o3
debug: CFLAGS+=-g -DDEBUG

libs: $(OBJ) $(LIBCUMATDIR)lib/libcumatrix.a
$(LIBCUMATDIR)lib/libcumatrix.a: $(OBJ)
	rm -f $@
	ar rcs $@ $^
	ranlib $@

vpath %.h include/
vpath %.cpp src/

INCLUDE=\
	-I include/\
	-I $(LIBCUMATDIR)include/\
	-I $(CUDA_DIR)include/\
	-I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L $(CUDA_DIR)lib64
LIBRARY=-lcuda -lcublas -lcudart
TARGET=test.app

all: matMultTest.cpp
	g++ -o $(TARGET) matMultTest.cpp $(OBJ) $(INCLUDE) $(LD_LIBRARY) $(LIBRARY)

debug: temp.cpp
	g++ -o temp.app temp.cpp $(OBJ) $(INCLUDE) $(LD_LIBRARY) $(LIBRARY)

clean:
	@rm -f $(TARGET) *o
