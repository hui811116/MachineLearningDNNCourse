CC=gcc
CXX=g++
CFLAGS= 
NVCC=nvcc -arch=sm_21 -w

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=hui
LIBCUMATDIR=tool/libcumatrix/
OBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/sigmoid.o

# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean o3

libs=$(LIBCUMATDIR)lib/libcumatrix.a

o3: CFLAGS+=-o3
o3: all

debug: CFLAGS+=-g -DDEBUG


#$(LIBCUMATDIR)lib/libcumatrix.a: $(OBJ)
#	rm -f $@
#	ar rcs $@ $^
#	ranlib $@

vpath %.h include/
vpath %.cpp src/

INCLUDE= -I include\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix
CPPFLAGS= -std=c++0x $(CFLAGS) $(INCLUDE)
TARGET=test.app

#<<<<<<< HEAD
#all: $(OBJ) $(HEADEROBJ) matMultTest.cpp
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LD_LIBRARY) $(LIBRARY)

#debug: $(OBJ) $(HEADEROBJ) temp.cpp
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY) $(LD_LIBRARY) 
#=======
all: $(OBJ) $(HEADEROBJ) $(EXECUTABLES)
	$(NVCC) $(INCLUDE) -o $@ $^ $(OBJ) $(LD_LIBRARY) $(LIBRARY)

debug: $(OBJ) $(HEADEROBJ) temp.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(OBJ) $(LIBRARY) $(LD_LIBRARY)

hui: matMultTest.cu $(libs)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(INCLUDE) -o hui.app $^ $(LD_LIBRARY) $(LIBRARY)
#>>>>>>> 0575673311a3d8bab804965375214e0ed60aa639

clean:
	@rm -f $(EXECUTABLES) obj/*

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $^

obj/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(INCLUDE) -o $@ -c $<
