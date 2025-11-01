CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

BUILD = build

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, $(BUILD)/%.o, $(SRC))
DEPS = $(OBJ:.o=.d)

TARGET = $(BUILD)/main

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

build:
	mkdir -p build

.PHONY: clean
clean:
	rm -rf $(BUILD)

-include $(wildcard $(DEPS))