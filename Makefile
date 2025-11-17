CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp

BUILD = build
BENCH = build/bench

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

BMSRC = $(wildcard bench/*.cpp)
BMOBJ = $(patsubst bench/%.cpp, $(BENCH)/%.o, $(BMSRC))
BMTARGETS = $(patsubst bench/%.cpp, $(BENCH)/%, $(BMSRC))
BMNAMES = $(notdir $(BMTARGETS))
BMFLAGS = -lbenchmark -lpthread

TARGET = $(BUILD)/main

.PHONY: all bench clean $(BMNAMES)

build:
	@mkdir -p build

build/bench:
	@mkdir -p build/bench

all: $(TARGET)

$(TARGET): $(OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cpp | build
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

# --------------- BENCHMARKING ----------------

$(BENCH)/%: $(BENCH)/%.o | build build/bench
	@echo "Linking benchmark $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BMFLAGS)

$(BENCH)/%.o: bench/%.cpp | build build/bench
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

bench: $(BMTARGETS) | build build/bench
	@echo "Running all benchmarks..."
	@for bm in $(BMTARGETS); do \
		echo "Executing $$bm"; \
		$$bm; \
	done

$(BMNAMES): %: $(BENCH)/% | build build/bench
	@echo "Running benchmark $@"
	@$(BENCH)/$@

clean:
	rm -rf $(BUILD)

-include $(wildcard $(DEPS))