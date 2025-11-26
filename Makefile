.DEFAULT_GOAL := all

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp

BUILD = build

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

# --------------- TESTING ----------------

TESTS = build/tests

TESTSRC = $(wildcard tests/*.cpp)
TESTOBJ = $(patsubst tests/%.cpp, $(TESTS)/%.o, $(TESTSRC))
TESTTARGETS = $(patsubst tests/%.cpp, $(TESTS)/%, $(TESTSRC))
TESTNAMES = $(notdir $(TESTTARGETS))
TESTFLAGS = -lgtest_main -lgtest -lpthread

# --------------- BENCHMARKING ----------------

BENCH = build/bench

BMSRC = $(wildcard bench/*.cpp)
BMOBJ = $(patsubst bench/%.cpp, $(BENCH)/%.o, $(BMSRC))
BMTARGETS = $(patsubst bench/%.cpp, $(BENCH)/%, $(BMSRC))
BMNAMES = $(notdir $(BMTARGETS))
BMFLAGS = -lbenchmark -lpthread
BMEXEFLAGS = --benchmark_repetitions=10 --benchmark_display_aggregates_only=true \
		--benchmark_out=results/benchmark_results.json --benchmark_out_format=json \
		--benchmark_counters_tabular=true

TARGET = $(BUILD)/main

.PHONY: all bench $(BMNAMES) tests $(TESTNAMES) clean

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
	$(CXX) $(CXXFLAGS) -o $@ $^ $(TESTFLAGS)

$(TESTS)/%.o: tests/%.cpp | build build/tests
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

tests: $(TESTTARGETS) | build build/tests
	@echo "Running all tests..."
	@for test in $(TESTTARGETS); do \
		echo "Executing $$test"; \
		$$test; \
	done

$(TESTNAMES): %: $(TESTS)/% | build build/tests
	@echo "Running test $@"
	@$(TESTS)/$@

# --------------- BENCHMARKING ----------------

$(BENCH)/%: $(BENCH)/%.o | build build/bench
	@echo "Linking benchmark $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BMFLAGS)

$(BENCH)/%.o: bench/%.cpp | build build/bench
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

bench: $(BMTARGETS) | build build/bench results
	@echo "Running all benchmarks..."
	@for bm in $(BMTARGETS); do \
		echo "Executing $$bm"; \
		$$bm $(BMEXEFLAGS); \
	done

$(BMNAMES): %: $(BENCH)/% | build build/bench results
	@echo "Running benchmark $@"
	@$(BENCH)/$@ $(BMEXEFLAGS)

clean:
	rm -rf $(BUILD)

-include $(wildcard $(DEPS))