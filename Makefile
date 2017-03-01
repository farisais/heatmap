CC?=gcc
CXX?=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc 

ARCH= -gencode arch=compute_20,code=[sm_20,sm_21] \
      -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
AR?=ar
GPU=1
COMMON=

# Release mode (If just dropping the lib into your project, check out -flto too.)
FLAGS=-fPIC -Wall -Wextra -I. -O3 -g -DNDEBUG -fopenmp
LDFLAGS=-fopenmp -O3 -lm

CFLAGS=$(FLAGS) -pedantic
CXXFLAGS=$(FLAGS) -std=c++0x

OBJS=heatmap.o $(patsubst %.c,%.o,$(wildcard colorschemes/*.c))

ifeq ($(GPU), 1) 
CFLAGS+= -DGPU
COMMON+= -DGPU -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
OBJS+= render.o
endif

# Debug mode
# FLAGS=-fPIC -Wall -Wextra -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -I. -O0 -g -fopenmp -Wa,-ahl=$(@:.o=.s)
# LDFLAGS=-fopenmp -O0 -g -lm
# TODO: Play with -D_GLIBCXX_PARALLEL
# TODO: Play with -D_GLIBCXX_PROFILE

# Also generate .s assembly output file:
# FLAGS=$(FLAGS) -Wa,-ahl=$(@:.o=.s)


.PHONY: all benchmarks samples clean

all: libheatmap.a libheatmap.so benchmarks examples tests
tests: tests/test
benchmarks: benchs/add_point_with_stamp benchs/weighted_unweighted benchs/rendering
examples: examples/heatmap_gen examples/heatmap_gen_weighted examples/simplest_cpp examples/simplest_c examples/huge examples/customstamps examples/customstamp_heatmaps examples/show_colorschemes

clean:
	rm -f libheatmap.a
	rm -f libheatmap.so
	rm -f render.o
	rm -f benchs/add_point_with_stamp
	rm -f benchs/rendering
	rm -f examples/heatmap_gen
	rm -f examples/heatmap_gen_weighted
	rm -f examples/simplest_c
	rm -f examples/simplest_cpp
	rm -f examples/simplest_libpng_cpp
	rm -f examples/huge
	rm -f examples/customstamps
	rm -f examples/customstamp_heatmaps
	rm -f examples/show_colorschemes
	rm -f tests/test
	find . -name '*.[os]' -print0 | xargs -0 rm -f

test: tests
	tests/test

heatmap.o: heatmap.c heatmap.h
	$(CC) -c $< $(CFLAGS) -o $@

render.o: render_kernel.cu
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

colorschemes/%.o: colorschemes/%.c colorschemes/%.h
	$(CC) -c $< $(CFLAGS) -o $@

libheatmap.a: $(OBJS)
	$(AR) rs $@ $^

libheatmap.so: $(OBJS)
	$(CC) $(LDFLAGS) -shared -o $@ $^

tests/test.o: tests/test.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

tests/test: tests/test.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/lodepng_cpp.o: examples/lodepng.cpp examples/lodepng.h
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/lodepng_c.o: examples/lodepng.cpp examples/lodepng.h
	$(CC) -x c -c $< $(CFLAGS) -o $@

examples/heatmap_gen.o: examples/heatmap_gen.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/heatmap_gen: examples/heatmap_gen.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/heatmap_gen_weighted.o: examples/heatmap_gen.cpp
	$(CXX) -c $< $(CXXFLAGS) -DWEIGHTED -o $@

examples/heatmap_gen_weighted: examples/heatmap_gen_weighted.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/simplest_cpp.o: examples/simplest.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/simplest_cpp: examples/simplest_cpp.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/simplest_c.o: examples/simplest.c
	$(CC) -c $< $(CFLAGS) -o $@

examples/simplest_c: examples/simplest_c.o examples/lodepng_c.o libheatmap.a
	$(CC) $^ $(LDFLAGS) -o $@

examples/simplest_libpng_cpp.o: examples/simplest_libpng.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/simplest_libpng_cpp: examples/simplest_libpng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -lpng -o $@

examples/huge.o: examples/huge.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/huge: examples/huge.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/customstamps.o: examples/customstamps.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/customstamps: examples/customstamps.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/customstamp_heatmaps.o: examples/customstamp_heatmaps.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/customstamp_heatmaps: examples/customstamp_heatmaps.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

examples/show_colorschemes.o: examples/show_colorschemes.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

examples/show_colorschemes: examples/show_colorschemes.o examples/lodepng_cpp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

benchs/add_point_with_stamp.o: benchs/add_point_with_stamp.cpp benchs/common.hpp benchs/timing.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

benchs/add_point_with_stamp: benchs/add_point_with_stamp.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

benchs/weighted_unweighted.o: benchs/weighted_unweighted.cpp benchs/common.hpp benchs/timing.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

benchs/weighted_unweighted: benchs/weighted_unweighted.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@

benchs/rendering.o: benchs/rendering.cpp benchs/common.hpp benchs/timing.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

benchs/rendering: benchs/rendering.o libheatmap.a
	$(CXX) $^ $(LDFLAGS) -o $@
