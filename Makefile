CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Werror -pedantic -Iinclude -O2
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/vector.c $(SRC_DIR)/rk4.c $(SRC_DIR)/cstr.c
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

TARGET = reactor_core
TEST_TARGET = test_runner

.PHONY: all test run clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(BUILD_DIR)/main.o
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_TARGET): $(OBJS) $(TEST_DIR)/test_solver.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

test: $(TEST_TARGET)
	@echo "========================================="
	@echo "Running Automated Unit Tests..."
	@echo "========================================="
	./$(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET) *.csv
