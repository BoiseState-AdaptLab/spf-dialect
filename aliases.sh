#!/bin/bash

export LLVM_SYMBOLIZER_PATH="../llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build standalone-opt Runtime"
alias rd="ASAN_OPTIONS=detect_leaks=0 build/bin/standalone-opt --my-pass --debug-only=my-pass dense_mttkrp_test.mlir"
alias rs="ASAN_OPTIONS=detect_leaks=0 build/bin/standalone-opt --my-pass --debug-only=my-pass sparse_mttkrp_test.mlir"
