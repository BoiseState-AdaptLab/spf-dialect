// Parts of this code are taken from
// https://github.com/llvm/llvm-project/blob/b682616d1fd1263b303985b9f930c1760033af2c/mlir/lib/ExecutionEngine/SparseTensorUtils.cpp
// Which is part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.

#include "Runtime/runtime.h"
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

extern "C" {

void _mlir_ciface_coords_gpu(StridedMemRefType<index_type, 1> *ref, void *coo,
                             index_type dim) {

  uint64_t *d_coords;
  uint64_t size;
  {
    std::vector<index_type> &v = static_cast<COO *>(coo)->coord[dim];
    size = v.size();
    cudaMalloc(&d_coords, sizeof(uint64_t) * size);
    cudaMemcpy(d_coords, v.data(), size * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    // zero it out just to make sure we're not doing computations with the wrong
    // thing.
    std::fill(v.begin(), v.end(), 0.0);
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

    // zero it out just to make sure we're not doing computations with the wrong
    // thing.
    std::fill(v.begin(), v.end(), 0.0);
  }

  ref->basePtr = ref->data = d_values;
  ref->offset = 0;
  ref->sizes[0] = size;
  ref->strides[0] = 1;
}

} // extern "C"