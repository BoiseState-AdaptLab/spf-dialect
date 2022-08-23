#!/bin/bash
set -euo pipefail

cat dense_matvec.mlir \
  | build/bin/standalone-opt \
    --linalg-generalize-named-ops \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-memref-to-llvm  \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
      | ../llvm-project/build/bin/mlir-cpu-runner \
        --entry-point-result=void \
        --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so,../llvm-project/build/lib/libmlir_c_runner_utils.so