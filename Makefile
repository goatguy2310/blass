.DEFAULT_GOAL := all

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp

BUILD = build

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

OS := $(shell uname)

# --------------- TESTING ----------------

TESTS = build/tests

TEST_SRC = $(wildcard tests/*.cpp)
TEST_OBJ = $(patsubst tests/%.cpp, $(TESTS)/%.o, $(TEST_SRC))
TEST_TARGETS = $(patsubst tests/%.cpp, $(TESTS)/%, $(TEST_SRC))
TEST_NAMES = $(notdir $(TEST_TARGETS))
TEST_FLAGS = -lgtest_main -lgtest -lpthread

# --------------- BENCHMARKING ----------------

BENCH = build/bench

BM_SRC = $(wildcard bench/*.cpp)
BM_OBJ = $(patsubst bench/%.cpp, $(BENCH)/%.o, $(BM_SRC))
BM_TARGETS = $(patsubst bench/%.cpp, $(BENCH)/%, $(BM_SRC))
BM_NAMES = $(notdir $(BM_TARGETS))
BM_FLAGS = -lbenchmark -lpthread
BM_EXE_FLAGS = --benchmark_display_aggregates_only=true \
		--benchmark_out=results/benchmark_results.json --benchmark_out_format=json \
		--benchmark_min_time=3s --benchmark_counters_tabular=true

# set to 1 to use all cores, 0 to pin to a single core
ALL_CORES ?= 0

# try linking with libpfm
HAS_LIBPFM := $(shell echo "int main() { return 0; }" | $(CXX) -x c++ - -lpfm -o /dev/null 2>/dev/null && echo 1 || echo 0)
ENABLED_PFM := 0

BM_EXEC_CMD = 
BM_CPU_CORE = "ALL"

ifeq ($(OS),Windows_NT)
	BM_EXE_FLAGS += --benchmark_repetitions=10
else
	# Linux/Unix Setup
	ifeq ($(HAS_LIBPFM),1)
		BM_FLAGS += -lpfm
		BM_EXE_FLAGS += --benchmark_perf_counters=cycles,instructions,l1-dcache-load-misses
		ENABLED_PFM := 1
	endif

	ifeq ($(ALL_CORES),1)
		# Case: Use All Cores (Standard Execution)
		BM_EXEC_CMD = 
	else
		# Case: Single Core (Pinning with taskset)
		# Logic: Find the best core (ignoring hyperthreading, sort by frequency)
		BM_CPU_CORE := $(shell lscpu -e=CPU,CORE,MAXMHZ | \
			grep -v "CPU" | \
			sort -k3,3nr -k2,2nr -k1,1n | \
			awk '$$2 != 0' | \
			awk '!seen[$$2]++' | \
			head -n 1 | \
			awk '{print $$1}')
		
		# Run with high priority on the pinned core
		BM_EXEC_CMD = sudo nice -n -20 taskset -c $(BM_CPU_CORE)
	endif
endif


TARGET = $(BUILD)/main

.PHONY: all bench $(BM_NAMES) tests $(TEST_NAMES) clean

all: $(TARGET)

build:
	@mkdir -p build

build/bench:
	@mkdir -p build/bench

build/tests:
	@mkdir -p build/tests

results:
	@mkdir -p results

$(TARGET): $(OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cpp | build
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

# --------------- TESTING ----------------

$(TESTS)/%: $(TESTS)/%.o | build build/tests
	@echo "Linking test $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(TEST_FLAGS)

$(TESTS)/%.o: tests/%.cpp | build build/tests
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

tests: $(TEST_TARGETS) | build build/tests
	@echo "Running all tests..."
	@for test in $(TEST_TARGETS); do \
		echo "Executing $$test"; \
		$$test; \
	done

$(TEST_NAMES): %: $(TESTS)/% | build build/tests
	@echo "Running test $@"
	@$(TESTS)/$@

# --------------- BENCHMARKING ----------------

$(BENCH)/%: $(BENCH)/%.o | build build/bench
	@echo "Linking benchmark $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BM_FLAGS)

$(BENCH)/%.o: bench/%.cpp | build build/bench
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

bench: $(BM_TARGETS) | build build/bench results
	@echo "Running all benchmarks (Cores: $(BM_CPU_CORE))..."
	@for bm in $(BM_TARGETS); do \
		echo "Executing $$bm"; \
		if [ "$(OS)" != "Windows_NT" ] && [ "$(ENABLED_PFM)" = "0" ]; then \
			echo "WARNING: libpfm not found. Perf counters will be disabled."; \
		fi; \
		$(BM_EXEC_CMD) $$bm $(BM_EXE_FLAGS); \
	done

$(BM_NAMES): %: $(BENCH)/% | build build/bench results
	@echo "Running benchmark $@ (Cores: $(BM_CPU_CORE))..."
	@if [ "$(OS)" != "Windows_NT" ] && [ "$(ENABLED_PFM)" = "0" ]; then \
		echo "WARNING: libpfm not found. Perf counters will be disabled."; \
	fi; \
	$(BM_EXEC_CMD) $(BENCH)/$@ $(BM_EXE_FLAGS)

clean:
	rm -rf $(BUILD)

-include $(wildcard $(DEPS))