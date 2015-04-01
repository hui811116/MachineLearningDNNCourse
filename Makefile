CC=gcc
CXX=g++
CFLAGS= 
NVCC=nvcc -arch=sm_21 -w

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=
LIBCUMATDIR=tool/libcumatrix/
OBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
CUMATOBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/sigmoid.o obj/dnn.o obj/dataset.o obj/datasetJason.o obj/parser.o

# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean o3

LIBS=$(LIBCUMATDIR)lib/libcumatrix.a

$(LIBCUMATDIR)lib/libcumatrix.a:$(CUMATOBJ)
	@echo "something wrong in tool/libcumatrix..."

DIR:
	@mkdir -p obj

o3: CFLAGS+=-o3
o3: all

debug: CFLAGS+=-g -DDEBUG

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include/\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix
CPPFLAGS= -O2 -std=c++11 $(CFLAGS) $(INCLUDE)
TARGET=test.app

#all:$(DIR) $(OBJ) $(HEADEROBJ) $(EXECUTABLES)
#	$(NVCC) $(INCLUDE) -o $@ $^ $(OBJ) $(LD_LIBRARY) $(LIBRARY)

#larry: $(OBJ) $(HEADEROBJ) temp.cpp
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBRARY) $(LD_LIBRARY)

#hui:$(HEADEROBJ) matMultTest.cu
#	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -o hui.app $(INCLUDE) $^ $(LIBS) $(LD_LIBRARY) $(LIBRARY)

train: $(OBJ) $(HEADEROBJ) train.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o train.app $^ $(LIBRARY) $(LD_LIBRARY)

#Pan: $(OBJ) $(HEADEROBJ) datasetTest.cpp 
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBRARY) $(LD_LIBRARY) 

CSV: $(OBJ) $(HEADEROBJ) testPredictSecond.cpp 
	$(CXX) $(CFLAGS) $(INCLUDE) -o CSV2.app $^ $(LIBRARY) $(LD_LIBRARY) 

clean:
	@rm -f $(EXECUTABLES) obj/* ./*.app

#jason: $(OBJ) $(HEADEROBJ) jasonTest.cpp
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBRARY) $(LD_LIBRARY)
# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

obj/datasetJason.o: src/datasetJason.cpp include/dataset.h 
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
obj/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
