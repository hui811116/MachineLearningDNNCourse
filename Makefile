CC=gcc
CXX=g++
CFLAGS= 
NVCC=nvcc -arch=sm_21 -w

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=hui
LIBCUMATDIR=tool/libcumatrix/
OBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
CUMATOBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ= obj/sigmoid.o
LARRYOBJ = obj/*.o

# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean o3

LIBS=$(LIBCUMATDIR)lib/libcumatrix.a

$(LIBCUMATDIR)lib/libcumatrix.a:$(CUMATOBJ)
	@echo "something wrong in tool/libcumatrix..."

o3: CFLAGS+=-o3
o3: all

debug: CFLAGS+=-g -DDEBUG

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix
CPPFLAGS= -std=c++0x $(CFLAGS) $(INCLUDE)
TARGET=test.app

all: $(OBJ) $(HEADEROBJ) $(EXECUTABLES)
	$(NVCC) $(INCLUDE) -o $@ $^ $(OBJ) $(LD_LIBRARY) $(LIBRARY)

debug: temp.cpp $(LIBS) $(LARRYOBJ)	
	g++ -std=c++0x -I include/ -I tool/libcumatrix/include/ -I /usr/local/cuda/samples/common/inc/ -I /usr/local/cuda/include/ -o temp.app obj/dnn.o obj/sigmoid.o tool/libcumatrix/lib/libcumatrix.a temp.cpp -L/usr/local/cuda/lib64/ -lcuda -lcublas -lcudart

hui:$(HEADEROBJ) matMultTest.cu $(LIBS)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(INCLUDE) -o hui.app $^ $(LD_LIBRARY) $(LIBRARY)

clean:
	@rm -f $(EXECUTABLES) obj/*

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $^
#	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $^

obj/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(INCLUDE) -o $@ -c $<
