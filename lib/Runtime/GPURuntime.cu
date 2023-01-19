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

void allocateAndPopulateGpuMemref(
    StridedMemRefType<double, 2> *ref, uint64_t dim1, uint64_t dim2,
    std::function<double(uint64_t i, uint64_t j)> &&fill) {
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

void allocateAndPopulateGpuMemref(StridedMemRefType<uint64_t, 1> *ref,
                                  uint64_t dim1,
                                  std::function<uint64_t(uint64_t _)> &&fill) {
  uint64_t *d_data;
  uint64_t size = dim1;

  std::vector<uint64_t> data = std::vector<uint64_t>(size);
  for (uint64_t i = 0; i < dim1; i++) {
    data[i] = fill(i);
  }

  cudaMalloc(&d_data, sizeof(uint64_t) * size);
  cudaMemcpy(d_data, data.data(), size * sizeof(uint64_t),
             cudaMemcpyHostToDevice);

  ref->basePtr = ref->data = d_data;
  ref->offset = 0;
  ref->sizes[0] = dim1;
  ref->strides[0] = 1;
}

DataForGpuMttkrp::DataForGpuMttkrp(char *filename, uint64_t argJ)
    : bData((COO *)_mlir_ciface_read_coo(filename)), NNZ(bData->nnz),
      I(bData->dims[0]), J(argJ), K(bData->dims[1]), L(bData->dims[2]) {
  assert(bData->rank == 3 && "mttkrp requires rank 3 tensor");

  // read data from b into gpu memory and construct memref into said data
  _mlir_ciface_coords_gpu(&bCoord0, bData, 0);
  _mlir_ciface_coords_gpu(&bCoord1, bData, 1);
  _mlir_ciface_coords_gpu(&bCoord2, bData, 2);
  _mlir_ciface_values_gpu(&bValues, bData);

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
  delete bData;
  cudaFree(bCoord0.data);
  cudaFree(bCoord1.data);
  cudaFree(bCoord2.data);
  cudaFree(c.data);
  cudaFree(d.data);
  cudaFree(a.data);
}

DataForGpuTTM::DataForGpuTTM(char *filename, uint64_t constantMode,
                             uint64_t inR)
    : xData((COO *)_mlir_ciface_read_coo(filename)), I(xData->dims[0]),
      J(xData->dims[1]), K(xData->dims[2]), R(inR) {
  assert(xData->rank == 3 && "ttm only supports rank 3 tensor");
  assert(constantMode < 3 && "constant mode dimension not in bounds");

  uint64_t constantModeDimSize = xData->dims[constantMode];

  // Sort data lexigraphically with constant mode considered the last. We need
  // to do this to be able to calculate fibers.
  xData->sortIndicesModeLast(constantMode);

  // read data from x into gpu memory and construct memref into said data
  _mlir_ciface_coords_gpu(&xCoordConstant, xData, constantMode);
  _mlir_ciface_values_gpu(&xValues, xData);

  // construct fptr vector
  auto fptrData = fiberStartStopIndices(*xData, constantMode);
  allocateAndPopulateGpuMemref(
      &fptr, fptrData.size(),
      [&](uint64_t i) -> uint64_t { return fptrData[i]; });

  Mf = fptrData.size() - 1;

  // Construct u matrix (constantModeDimSize x R)
  // 2x2 example: [[0,1],
  //               [2,3]]
  allocateAndPopulateGpuMemref(&u, constantModeDimSize, R,
                               [=](uint64_t i, uint64_t j) -> double {
                                 return i * constantModeDimSize + j;
                               });

  // construct y matrix
  //
  // y is stored "semi-sparsely" which meas that dense fibers are stored at
  // sparse coordinates.
  //
  // Ex:
  //  this semi-sparse matrix:
  //      ndims:
  //      2x3x4
  //      inds:
  //      sptIndexVector length: 11
  //      0       0       0       1       1       1       1       2       2 2
  //      2 sptIndexVector length: 11 0       2       3       0       1 2 3 0
  //      1       2       3 values: 11 x 2 matrix 154.00  231.00 20.00   33.00
  //      92.00   201.00
  //      122.00  183.00
  //      106.00  170.00
  //      6.00    109.00
  //      150.00  225.00
  //      0.00    66.00
  //      44.00   127.00
  //      36.00   67.00
  //      0.00    43.00
  //  corresponds to this dense matrix:
  //      [[[154.   0.  20.  92.]
  //        [122. 106.   6. 150.]
  //        [  0.  44.  36.   0.]]
  //
  //       [[231.   0.  33. 201.]
  //        [183. 170. 109. 225.]
  //        [ 66. 127.  67.  43.]]]
  allocateAndPopulateGpuMemref(
      &y, Mf, R, [](uint64_t i, uint64_t j) -> double { return 0; });
}

DataForGpuTTM::~DataForGpuTTM() {
  delete xData;
  cudaFree(fptr.data);
  cudaFree(xCoordConstant.data);
  cudaFree(xValues.data);
  cudaFree(u.data);
  cudaFree(y.data);
}
