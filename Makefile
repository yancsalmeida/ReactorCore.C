CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Werror -pedantic -mavx2 -mfma -Iinclude -O3 -fPIC
ifdef AVX512
  CFLAGS += -mavx512f -mavx512dq
endif
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/vector.c $(SRC_DIR)/rk4.c $(SRC_DIR)/cstr.c $(SRC_DIR)/reactor_ffi.c
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

TARGET = reactor_core
TEST_TARGET = test_runner
SHARED_LIB = libreactorcore.so

.PHONY: all test run clean lib

all: $(TARGET) $(SHARED_LIB)

lib: $(SHARED_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SHARED_LIB): $(OBJS)
	$(CC) -shared $(OBJS) $(LDFLAGS) -o $@

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
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET) $(SHARED_LIB) *.csv
