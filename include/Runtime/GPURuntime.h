#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <vector>

// Copies data and returns the vector that owns the backing buffer for the
// memref. There is an OwningMemref we could create and return type but it uses
// some C++ features not available in the CUDA compiler I'm using, and it
// appears you can only use the cudaMalloc etc. functions inside .cu files. This
// is good enough ¯\_(ツ)_/¯.
template <typename T, int N>
std::vector<T> copyToCpuMemRef(StridedMemRefType<T, N> *srcGpuMemRef,
                               StridedMemRefType<T, N> *destCpuMemRef);

extern "C" {
void _mlir_ciface_coords_gpu(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                             uint64_t dim);

void _mlir_ciface_values_gpu(StridedMemRefType<double, 1> *ref, void *coo);
} // extern "C"

#endif // GPU_RUNTIME_H