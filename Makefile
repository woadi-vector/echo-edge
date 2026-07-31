CC      ?= cc
INC     := -Icore -Ibench
COMMON  := -std=c99 -Wall -Wextra $(INC)

# Baseline: no optimization. This is the honest "before" column.
BASE_FLAGS := -O0

# Optimized: aggressive scalar + autovectorization.
OPT_FLAGS  := -O3 -ffast-math -funroll-loops

# Arm-specific: tuned for Neoverse N2 (Azure Cobalt 100). Override ARM_MCPU
# for other targets, e.g. ARM_MCPU=neoverse-n1 on Ampere Altra.
ARM_MCPU   ?= neoverse-n2
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
OPT_FLAGS += -mcpu=$(ARM_MCPU)
endif

SRC := core/echo.c bench/bench.c

.PHONY: all model bench baseline optimized clean compare

all: optimized

model:
	python3 train/train.py

baseline: core/echo_model.h
	$(CC) $(COMMON) $(BASE_FLAGS) $(SRC) -lm -o bench-baseline

optimized: core/echo_model.h
	$(CC) $(COMMON) $(OPT_FLAGS) $(SRC) -lm -o bench-optimized

bench: optimized
	./bench-optimized

# The submission table: same source, same machine, two flag sets.
compare: baseline optimized
	@echo "=== BASELINE (-O0) ==="; ./bench-baseline
	@echo; echo "=== OPTIMIZED ($(OPT_FLAGS)) ==="; ./bench-optimized

core/echo_model.h:
	python3 train/train.py

clean:
	rm -f bench-baseline bench-optimized wasm/echo.js wasm/echo.wasm
