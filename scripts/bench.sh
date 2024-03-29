#!/bin/bash
set -euo pipefail

make -C ../build bench

# export ITERATIONS=1

platforms=("cpu" "gpu")
benchmarks=("mttkrp" "ttm")
implementations=("mlir" "iegenlib")

for file in "$@"; do
	for platform in "${platforms[@]}"; do
		for benchmark in "${benchmarks[@]}"; do
			for implementation in "${implementations[@]}"; do
				../build/bench/bench "$platform" "$benchmark" "$implementation" "$file"
			done
		done
	done
done
