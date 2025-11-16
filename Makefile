CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp

BUILD = build

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

BMSRC = $(wildcard bench/*.cpp)
BMOBJ = $(patsubst bench/%.cpp, $(BUILD)/%.o, $(BMSRC))
BMTARGET = $(BUILD)/benchmark
BMFLAGS = -lbenchmark -lpthread

TARGET = $(BUILD)/main

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BMTARGET): $(BMOBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ -lbenchmark -lpthread

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
bench: $(BMTARGET)
	./$(BMTARGET)

-include $(wildcard $(DEPS))