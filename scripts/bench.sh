#!/bin/bash
set -euo pipefail

make -C ../build bench

platforms=("cpu" "gpu")
benchmarks=("mttkrp")
implementations=("mlir" "iegenlib" )

for platform in "${platforms[@]}"; do
	for benchmark in "${benchmarks[@]}"; do
		for implementation in "${implementations[@]}"; do
			for file in "$@"; do
				../build/bench/bench "$platform" "$benchmark" "$implementation" "$file"
			done
		done
	done
done