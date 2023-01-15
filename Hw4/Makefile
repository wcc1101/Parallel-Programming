CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm -pthread
CFLAGS = -O3 -lm -phread
TARGETS = mapreduce

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)