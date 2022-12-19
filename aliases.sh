#!/bin/bash

export LLVM_SYMBOLIZER_PATH="/home/aaronstgeorge/Dev/c++/llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build standalone-opt Runtime jacobi-c-print-example"
alias rd="build/bin/standalone-opt --my-pass --debug-only=my-pass dense_mttkrp_test.mlir"
alias rm="build/bin/standalone-opt --my-pass --debug-only=my-pass simple_multi_statement_test.mlir"
alias rt="./run_test.sh"
