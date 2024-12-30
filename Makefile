SRCS=./test/test.cpp
OBJDIR=./build
RESDIR=./results
OBJS = $(patsubst ./test/test.cpp,$(OBJDIR)/test.o,$(SRCS))
TARGET = maria
CXX := g++
MKLROOT = /usr/include/mkl
OPTION = -I./ -DIN_PARALLEL  -I /usr/include/eigen3 -fopenmp -march=native -ffast-math -flto -I$(MKLROOT) -DNDEBUG
LFLAGS = -std=c++11 -O3 $(OPTION)  -L$(MKLROOT)/intel64 -lboost_timer -lmkl_intel_lp64 -lmkl_core  -lmkl_gnu_thread -lpthread -lm -ldl
CXXFLAGS := -std=c++17 -mavx512f -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -mavx -mfma -mpopcnt
# CXXFLAGS := -O3 -I /usr/include/eigen3 -fopenmp -mcmodel=medium -std=c++17 -mcpu=native #-fpic -mavx512f -lrt -DHAVE_CXX0X -ftree-vectorize -ftree-vectorizer-verbose=0 -openmp -DNDEBUG 

.PHONY:maria

all: $(TARGET) 


maria:./test/maria.cpp
	@test -d ./indexes | mkdir -p ./indexes
	@test -d ./results | mkdir -p ./results
	$(CXX) $(CXXFLAGS) -o maria ./test/maria.cpp



clean:
	rm -rf $(TARGET) $(OBJDIR) maria
