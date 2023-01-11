// Parts of this code are taken from
// https://github.com/llvm/llvm-project/blob/b682616d1fd1263b303985b9f930c1760033af2c/mlir/lib/ExecutionEngine/SparseTensorUtils.cpp
// Which is part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.

#include "Runtime/CPURuntime.h"
#include "Runtime/GPURuntime.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <vector>

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

template <> std::vector<uint64_t> copyToCpuMemRef(StridedMemRefType<uint64_t, 1> *srcGpuMemRef, StridedMemRefType<uint64_t, 1> *destCpuMemRef) {
  auto size = srcGpuMemRef->sizes[0] ;
  std::vector<uint64_t> backingMemory(size);
  cudaMemcpy(backingMemory.data(), srcGpuMemRef->data,  size * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  destCpuMemRef->basePtr = destCpuMemRef->data = backingMemory.data();
  destCpuMemRef->offset = 0;
  destCpuMemRef->sizes[0] = size;
  destCpuMemRef->strides[0] = 1;

  return backingMemory;
}

template <>
std::vector<double> copyToCpuMemRef(StridedMemRefType<double, 1> *srcGpuMemRef, StridedMemRefType<double, 1> *destCpuMemRef) {
  auto size = srcGpuMemRef->sizes[0] ;
  std::vector<double> backingMemory(size);
  cudaMemcpy(backingMemory.data(), srcGpuMemRef->data,  size * sizeof(double),
             cudaMemcpyDeviceToHost);

  destCpuMemRef->basePtr = destCpuMemRef->data = backingMemory.data();
  destCpuMemRef->offset = 0;
  destCpuMemRef->sizes[0] = size;
  destCpuMemRef->strides[0] = srcGpuMemRef->strides[0]; // this will alwyas be 1

  return backingMemory;
}

template <> std::vector<double> copyToCpuMemRef(StridedMemRefType<double, 2> *srcGpuMemRef, StridedMemRefType<double, 2> *destCpuMemRef) {
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

void allocateAndPopulateGpuMemref(
    StridedMemRefType<double, 2> *ref, uint64_t dim1, uint64_t dim2,
    std::function<double(uint64_t i, uint64_t j)> fill) {
  double *d_data;
  uint64_t size = dim1 * dim2;

  std::vector<double> data = std::vector<double>(size);
  for (uint64_t i = 0; i < dim1; i++) {
    for (uint64_t j = 0; j < dim2; j++) {
      data[i * dim2 + j] = fill(i, j);
    }
  }

  cudaMalloc(&d_data, sizeof(double) * size);
  cudaMemcpy(d_data, data.data(), size * sizeof(double),
             cudaMemcpyHostToDevice);

  ref->basePtr = ref->data = d_data;
  ref->offset = 0;
  ref->sizes[0] = dim1;
  ref->sizes[1] = dim2;
  ref->strides[0] = dim2;
  ref->strides[1] = 1;
}

DataForGpuMttkrp::DataForGpuMttkrp(char *filename, uint64_t argJ)
    : coo((COO *)_mlir_ciface_read_coo(filename)), NNZ(coo->nnz),
      I(coo->dims[0]), J(argJ), K(coo->dims[1]), L(coo->dims[2]) {

  assert(coo->rank == 3 && "mttkrp requires rank 3 tensor");
  _mlir_ciface_coords_gpu(&bCoord0, coo, 0);
  _mlir_ciface_coords_gpu(&bCoord1, coo, 1);
  _mlir_ciface_coords_gpu(&bCoord2, coo, 2);
  _mlir_ciface_values_gpu(&bValues, coo);

  // 2x2 example: [[0,1],
  //               [2,3]]
  auto fillRowsIndrementing = [=](uint64_t i, uint64_t j) -> double {
    return i * J + j;
  };

  // Construct c matrix (K x J)
  allocateAndPopulateGpuMemref(&c, K, J, fillRowsIndrementing);

  // Construct d matrix (L x J)
  allocateAndPopulateGpuMemref(&d, L, J, fillRowsIndrementing);

  // Construct a matrix (I x J)
  allocateAndPopulateGpuMemref(
      &a, I, J, [](uint64_t i, uint64_t j) -> double { return 0; });
}

DataForGpuMttkrp::~DataForGpuMttkrp() {
  delete coo;
  cudaFree(bCoord0.data);
  cudaFree(bCoord1.data);
  cudaFree(bCoord2.data);
  cudaFree(c.data);
  cudaFree(d.data);
  cudaFree(a.data);
}
