#ifndef READ_DATA_GPU_H
#define READ_DATA_GPU_H

#include "Runtime/CPURuntime.h"
#include "benchmarks.h"
#include <cstdint>

class DataForGpuMttkrp {
  COO *bData;

public:
  StridedMemRefType<uint64_t, 1> bCoord0;
  StridedMemRefType<uint64_t, 1> bCoord1;
  StridedMemRefType<uint64_t, 1> bCoord2;
  StridedMemRefType<float, 1> bValues;
  StridedMemRefType<float, 2> c;
  StridedMemRefType<float, 2> d;
  StridedMemRefType<float, 2> a;
  const uint64_t NNZ;
  const uint64_t I;
  const uint64_t J;
  const uint64_t K;
  const uint64_t L;

  DataForGpuMttkrp(char *filename, Config config);
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
  StridedMemRefType<float, 1> xValues;
  StridedMemRefType<float, 2> u;
  StridedMemRefType<float, 2> y;
  const uint64_t constantMode;

  DataForGpuTTM(char *filename, Config config);
  ~DataForGpuTTM();
};

#endif // READ_DATA_GPU_H