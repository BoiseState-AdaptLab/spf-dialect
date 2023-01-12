#!/bin/bash
set -euo pipefail

export ASAN_OPTIONS=detect_leaks=0

echo  "EXPECTED OUTPUT ========================="
build/jacobi/jacobi-no-transform-c-print-example
echo  "OUTPUT =================================="
build/bin/standalone-opt no_transform_multi_statement_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
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
    --shared-libs=build/lib/Runtime/libCPURuntime.so \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so
echo  "EXPECTED OUTPUT ========================="
build/jacobi/jacobi-transformed-c-print-example
echo  "OUTPUT =================================="
build/bin/standalone-opt transformed_multi_statement_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
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
    --shared-libs=build/lib/Runtime/libCPURuntime.so \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so
echo  "EXPECTED OUTPUT ========================="
echo  "( ( 16075, 21930, 28505, 35800, 43815 ),"
echo  "  ( 10000, 14225, 19180, 24865, 31280 ) )"
echo  "DENSE-CPU================================"
build/bin/standalone-opt dense_mttkrp_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
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
echo  "SPARSE-CPU==============================="
build/bin/standalone-opt sparse_mttkrp_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
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
  | TENSOR0="mttkrp_b.tns" ../llvm-project/build/bin/mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=build/lib/Runtime/libCPURuntime.so \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so
echo  "DENSE-GPU================================"
build/bin/standalone-opt dense_mttkrp_gpu_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
  -gpu-map-parallel-loops \
  -convert-parallel-loops-to-gpu \
  -lower-affine \
  -convert-vector-to-scf \
  -convert-scf-to-cf \
  -func-bufferize \
  -arith-bufferize \
  -finalizing-bufferize \
  -gpu-kernel-outlining \
  | build/bin/standalone-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' \
  | build/bin/standalone-opt -gpu-to-llvm \
  -convert-vector-to-llvm \
  -convert-memref-to-llvm \
  -convert-complex-to-standard \
  -convert-math-to-llvm \
  -convert-complex-to-llvm \
  -convert-math-to-libm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | TENSOR0="mttkrp_b.tns" ../llvm-project/build/bin/mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=build/lib/Runtime/libCPURuntime.so \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_cuda_runtime.so
echo  "SPARSE-GPU==============================="
build/bin/standalone-opt sparse_mttkrp_gpu_test.mlir \
  -my-pass \
  -inline \
  -cse \
  -lower-affine \
  -gpu-map-parallel-loops \
  -convert-parallel-loops-to-gpu \
  -lower-affine \
  -convert-vector-to-scf \
  -convert-scf-to-cf \
  -func-bufferize \
  -arith-bufferize \
  -finalizing-bufferize \
  -gpu-kernel-outlining \
  | build/bin/standalone-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' \
  | build/bin/standalone-opt -gpu-async-region \
  -gpu-to-llvm \
  -convert-vector-to-llvm \
  -convert-memref-to-llvm \
  -convert-complex-to-standard \
  -convert-math-to-llvm \
  -convert-complex-to-llvm \
  -convert-math-to-libm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | TENSOR0="mttkrp_b.tns" ../llvm-project/build/bin/mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=build/lib/Runtime/libGPURuntime.so \
    --shared-libs=build/lib/Runtime/libCPURuntime.so \
    --shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_c_runner_utils.so \
    --shared-libs=../llvm-project/build/lib/libmlir_cuda_runtime.so