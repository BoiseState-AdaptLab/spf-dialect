#!/bin/bash

export LLVM_SYMBOLIZER_PATH="/home/aaronstgeorge/Dev/c++/llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build standalone-opt Runtime jacobi-c-print-example"
alias rmn="build/bin/standalone-opt --my-pass --cse --debug-only=my-pass no_transform_multi_statement_test.mlir"
alias rmt="build/bin/standalone-opt --my-pass --cse --debug-only=my-pass transformed_multi_statement_test.mlir"
alias rt="./run_test.sh"
