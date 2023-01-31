// Parts of this code are taken from
// https://github.com/llvm/llvm-project/blob/b682616d1fd1263b303985b9f930c1760033af2c/mlir/lib/ExecutionEngine/SparseTensorUtils.cpp
// Which is part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.

#include "Runtime/CPURuntime.h"
#include "Runtime/GPURuntime.h"

extern "C" {
void _mlir_ciface_coords_gpu(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                             uint64_t dim) {

  uint64_t *d_coords;
  uint64_t size;
  {
    std::vector<uint64_t> &v = static_cast<COO *>(coo)->coord[dim];
    size = v.size();
    cudaMalloc(&d_coords, sizeof(uint64_t) * size);
    cudaMemcpy(d_coords, v.data(), size * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
  }

  ref->basePtr = ref->data = d_coords;
  ref->offset = 0;
  ref->sizes[0] = size;
  ref->strides[0] = 1;
}

void _mlir_ciface_values_gpu(StridedMemRefType<double, 1> *ref, void *coo) {

  double *d_values;
  uint64_t size;
  {
    std::vector<double> &v = static_cast<COO *>(coo)->values;
    size = v.size();
    cudaMalloc(&d_values, sizeof(uint64_t) * size);
    cudaMemcpy(d_values, v.data(), size * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  ref->basePtr = ref->data = d_values;
  ref->offset = 0;
  ref->sizes[0] = size;
  ref->strides[0] = 1;
}
} // extern "C"

template <>
std::vector<uint64_t>
copyToCpuMemRef(StridedMemRefType<uint64_t, 1> *srcGpuMemRef,
                StridedMemRefType<uint64_t, 1> *destCpuMemRef) {
  auto size = srcGpuMemRef->sizes[0];
  std::vector<uint64_t> backingMemory(size);
  cudaMemcpy(backingMemory.data(), srcGpuMemRef->data, size * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  destCpuMemRef->basePtr = destCpuMemRef->data = backingMemory.data();
  destCpuMemRef->offset = 0;
  destCpuMemRef->sizes[0] = size;
  destCpuMemRef->strides[0] = 1;

  return backingMemory;
}

template <>
std::vector<double>
copyToCpuMemRef(StridedMemRefType<double, 1> *srcGpuMemRef,
                StridedMemRefType<double, 1> *destCpuMemRef) {
  auto size = srcGpuMemRef->sizes[0];
  std::vector<double> backingMemory(size);
  cudaMemcpy(backingMemory.data(), srcGpuMemRef->data, size * sizeof(double),
             cudaMemcpyDeviceToHost);

  destCpuMemRef->basePtr = destCpuMemRef->data = backingMemory.data();
  destCpuMemRef->offset = 0;
  destCpuMemRef->sizes[0] = size;
  destCpuMemRef->strides[0] = srcGpuMemRef->strides[0]; // this will alwyas be 1

  return backingMemory;
}

template <>
std::vector<double>
copyToCpuMemRef(StridedMemRefType<double, 2> *srcGpuMemRef,
                StridedMemRefType<double, 2> *destCpuMemRef) {
  auto size = srcGpuMemRef->sizes[0] * srcGpuMemRef->sizes[1];
  std::vector<double> backingMemory(size);
  cudaMemcpy(backingMemory.data(), srcGpuMemRef->data, size * sizeof(double),
             cudaMemcpyDeviceToHost);

  destCpuMemRef->basePtr = destCpuMemRef->data = backingMemory.data();
  destCpuMemRef->offset = 0;
  destCpuMemRef->sizes[0] = srcGpuMemRef->sizes[0];
  destCpuMemRef->sizes[1] = srcGpuMemRef->sizes[1];
  destCpuMemRef->strides[0] = srcGpuMemRef->strides[0];
  destCpuMemRef->strides[1] = srcGpuMemRef->strides[1];

  return backingMemory;
}
