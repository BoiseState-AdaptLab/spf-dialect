#!/bin/bash
set -euo pipefail

export ASAN_OPTIONS=detect_leaks=0

echo  "EXPECTED OUTPUT ========================="
echo  "( ( 16075, 21930, 28505, 35800, 43815 ),"
echo  "  ( 10000, 14225, 19180, 24865, 31280 ) )"
echo  "========================================="
build/bin/standalone-opt dense_mttkrp_test.mlir \
  -my-pass \
  -convert-vector-to-scf \
  -convert-scf-to-cf \
  -gpu-to-llvm \
  -convert-vector-to-llvm \
  -convert-memref-to-llvm \
  -convert-complex-to-standard \
  -convert-math-to-llvm \
  -convert-complex-to-llvm \
  -convert-math-to-libm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | ../llvm-project/build/bin/mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so
