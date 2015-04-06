CC=gcc
CXX=g++
CPPFLAGS= -O2 -std=c++11 $(INCLUDE)
NVCC=nvcc -arch=sm_21 -w

CUDA_DIR=/usr/local/cuda/

EXECUTABLES=
LIBCUMATDIR=tool/libcumatrix/
CUMATOBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/transforms.o obj/dnn.o obj/dataset.o obj/datasetJason.o obj/parser.o

# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean 

LIBS=$(LIBCUMATDIR)lib/libcumatrix.a

$(LIBCUMATDIR)lib/libcumatrix.a:
	@echo "Missing library file, trying to fix it in tool/libcumatrix"
	@cd tool/libcumatrix/ ; make ; cd ../..
debug: CPPFLAGS+=-g -DDEBUG

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include/\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix
TARGET=test.app

DIR=OBJDIR
OBJDIR:
	@mkdir -p obj

#all:$(DIR) $(HEADEROBJ) $(EXECUTABLES)

larry: $(HEADEROBJ) temp.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

hui: $(HEADEROBJ) matMultTest.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -o hui.app $(INCLUDE) $^ $(LIBS) $(LD_LIBRARY) $(LIBRARY)

train:  $(HEADEROBJ) train.cpp
	@echo "compiling train.app for DNN Training"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o train.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

#Pan: $(HEADEROBJ) datasetTest.cpp 
#	$(CXX) $(CFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY) 

CSV: $(HEADEROBJ) CSVTest.cpp 
	@echo "compiling CSV2.app for generating CSV format testing results"
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o CSV2.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY) 

clean:
	@echo "All objects and executables removed"
	@rm -f $(EXECUTABLES) obj/* ./*.app

jason: $(HEADEROBJ) jasonTest.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $(TARGET) $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

ctags:
	@rm -f src/tags tags
	@echo "Tagging src directory"
	@cd src; ctags -a *.cpp ../include/*.h; ctags -a *.cu ../include/*.h; cd ..
	@echo "Tagging main directory"
	@ctags -a *.cpp src/* ; ctags -a *.cu src/*
	
# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	@echo "compiling OBJ: $@ " 
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

obj/datasetJason.o: src/datasetJason.cpp include/dataset.h 
	@echo "compiling OBJ: $@ "
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
obj/%.o: %.cu
	@echo "compiling OBJ: $@ "
	@$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
