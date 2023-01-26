#!/bin/bash

export LLVM_SYMBOLIZER_PATH="/home/aaronstgeorge/Dev/c++/llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build standalone-opt CPURuntime GPURuntime jacobi-no-transform-c-print-example jacobi-transformed-c-print-example"
alias rmn="build/bin/standalone-opt --my-pass --cse --lower-affine --debug-only=my-pass no_transform_multi_statement_test.mlir"
alias rmt="build/bin/standalone-opt --my-pass --cse --lower-affine --debug-only=my-pass transformed_multi_statement_test.mlir"
alias rss="build/bin/standalone-opt --my-pass --cse --lower-affine --debug-only=my-pass sparse_mttkrp_test.mlir"
alias rsp="build/bin/standalone-opt --my-pass --cse --lower-affine --debug-only=my-pass sparse_mttkrp_gpu_test.mlir"
alias rj="build/bin/standalone-opt --my-pass --cse --lower-affine --debug-only=my-pass jacobi.mlir"
alias rt="./run_test.sh"
