#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include "Runtime/CPURuntime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <vector>

// Copies data and returns the vector that owns the backing buffer for the
// memref. There is an OwningMemref we could create and return type but it uses
// some C++ features not available in the CUDA compiler I'm using, and it
// appears you can only use the cudaMalloc etc. functions inside .cu files. This
// is good enough ¯\_(ツ)_/¯.
template <typename T, int N>
std::vector<T> copyToCpuMemRef(StridedMemRefType<T, N> *srcGpuMemRef, StridedMemRefType<T, N> *destCpuMemRef);

class DataForGpuMttkrp {
  COO *bData;

public:
  StridedMemRefType<uint64_t, 1> bCoord0;
  StridedMemRefType<uint64_t, 1> bCoord1;
  StridedMemRefType<uint64_t, 1> bCoord2;
  StridedMemRefType<double, 1> bValues;
  StridedMemRefType<double, 2> c;
  StridedMemRefType<double, 2> d;
  StridedMemRefType<double, 2> a;
  const uint64_t NNZ;
  const uint64_t I;
  const uint64_t J;
  const uint64_t K;
  const uint64_t L;

  DataForGpuMttkrp(char *filename, uint64_t argJ);
  ~DataForGpuMttkrp();
};

class DataForGpuTTM {
  COO *xData;

public:
  uint64_t Mf; // number of n-mode fibers
  const uint64_t I;
  const uint64_t J;
  const uint64_t K;
  const uint64_t R;
  StridedMemRefType<uint64_t, 1> fptr; // the beginnings of each X mode-n fiber
  StridedMemRefType<uint64_t, 1>
      xCoordConstant; // the coordinates in dimension <constantMode>
  StridedMemRefType<double, 1> xValues;
  StridedMemRefType<double, 2> u;
  StridedMemRefType<double, 2> y;

  DataForGpuTTM(char *filename, uint64_t constantMode, uint64_t inR);
  ~DataForGpuTTM();
};

#endif // GPU_RUNTIME_H