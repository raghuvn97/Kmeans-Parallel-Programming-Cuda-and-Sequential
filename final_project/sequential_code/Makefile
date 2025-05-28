# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -pg
TARGET := kmeans_seq
SRC := kmeans.cpp

# Default target
all: compile

# Compilation target
compile:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Run target
run: compile
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) gmon.out

# Phony targets (not files)
.PHONY: all compile run clean