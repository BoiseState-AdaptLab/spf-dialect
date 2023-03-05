#!/bin/bash

export LLVM_SYMBOLIZER_PATH="/home/aaronstgeorge/Dev/c++/llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build spf-opt CPURuntime GPURuntime jacobi-no-transform-c-print-example jacobi-transformed-c-print-example"
alias r="build/bin/spf-opt --my-pass --cse --debug-only=my-pass bench/sparse_mttkrp_cpu.mlir"
alias rt="./run_test.sh"
