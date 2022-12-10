#!/bin/bash

export LLVM_SYMBOLIZER_PATH="/home/aaronstgeorge/Dev/c++/llvm-project/build/bin/llvm-symbolizer"
alias b="make standalone-opt Runtime"
alias r="bin/standalone-opt --my-pass --debug-only=my-pass ../simple_multi_statement_test.mlir"
