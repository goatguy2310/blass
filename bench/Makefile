CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp

BUILD = build/bench

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

BMSRC = $(wildcard bench/*.cpp)
BMOBJ = $(patsubst bench/%.cpp, $(BENCH)/%.o, $(BMSRC))
BMTARGETS = $(patsubst bench/%.cpp, $(BENCH)/%, $(BMSRC))
BMFLAGS = -lbenchmark -lpthread

TARGET = $(BUILD)/main

.PHONY: all
all: $(BMTARGETS)

$(TARGET): $(OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# for each benchmark source file, create a corresponding target
$(BENCH)/%: $(BMOBJ)
	@echo "Linking benchmark $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BMFLAGS)

$(BUILD)/%.o: src/%.cpp | build
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/%.o: bench/%.cpp | build
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

build:
	@mkdir -p build
	@mkdir -p build/tensor
	@mkdir -p build/bench

.PHONY: clean
clean:
	rm -rf $(BUILD)

.PHONY: bench
bench: $(BMTARGETS)
	@echo "Running all benchmarks..."
	@for bm in $(BMTARGETS); do \
		echo "Executing $$bm"; \
		$$bm; \
	done

-include $(wildcard $(DEPS))