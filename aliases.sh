#!/bin/bash

export LLVM_SYMBOLIZER_PATH="../llvm-project/build/bin/llvm-symbolizer"
alias b="make -C build standalone-opt Runtime"
alias r="build/bin/standalone-opt --my-pass --debug-only=my-pass"
