#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <ratio>
#include <system_error>
#include <vector>

#include "Runtime/CPURuntime.h"
#include "benchmarks.h"
#include "read_data_cpu.h"

extern "C" {
int64_t _mlir_ciface_sparse_ttm_cpu(
    uint64_t Mf, uint64_t R, StridedMemRefType<uint64_t, 1> *fptr,
    StridedMemRefType<uint64_t, 1> *x_coord_constant,
    StridedMemRefType<float, 1> *x_values, StridedMemRefType<float, 2> *u,
    StridedMemRefType<float, 2> *y);
int64_t _mlir_ciface_sparse_mttkrp_cpu(
    uint64_t NNZ, uint64_t J, StridedMemRefType<uint64_t, 1> *b_coord_0,
    StridedMemRefType<uint64_t, 1> *b_coord_1,
    StridedMemRefType<uint64_t, 1> *b_coord_2,
    StridedMemRefType<float, 1> *b_values, StridedMemRefType<float, 2> *c,
    StridedMemRefType<float, 2> *d, StridedMemRefType<float, 2> *a);
int64_t _mlir_ciface_sparse_ttm_gpu(
    uint64_t Mf, uint64_t R, StridedMemRefType<uint64_t, 1> *fptr,
    StridedMemRefType<uint64_t, 1> *x_coord_constant,
    StridedMemRefType<float, 1> *x_values, StridedMemRefType<float, 2> *u,
    StridedMemRefType<float, 2> *y);
int64_t _mlir_ciface_sparse_mttkrp_gpu(
    uint64_t NNZ, uint64_t J, StridedMemRefType<uint64_t, 1> *b_coord_0,
    StridedMemRefType<uint64_t, 1> *b_coord_1,
    StridedMemRefType<uint64_t, 1> *b_coord_2,
    StridedMemRefType<float, 1> *b_values, StridedMemRefType<float, 2> *c,
    StridedMemRefType<float, 2> *d, StridedMemRefType<float, 2> *a);
}

namespace {
template <typename T>
void runTest(T &result, const char *context, char *filename, Config config) {
  auto red = "\033[31m";
  auto green = "\033[32m";
  auto reset = "\033[0m";
  auto reference = T(filename, config, /*isReference*/ true);
  if (!reference.isSame(result)) {
    std::cerr << red << context
              << " result doesn't match reference implementation\n"
              << reset;
    exit(1);
  } else {
    std::cerr << green << context
              << " result matches reference implementation\n"
              << reset;
  }
}
} // anonymous namespace

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> cpu_mttkrp_mlir(Config config, char *filename) {
  if (config.debug) {
    std::cout << "cpu mttkrp mlir =====\n";
  }

  DataForCpuMttkrp data(filename, config);

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    times[i] = _mlir_ciface_sparse_mttkrp_cpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (config.debug) {
    data.dump();
    std::cout << "=====\n";
  }

  if (config.test) {
    runTest(data, "cpu mttkrp mlir", filename, config);
  }

  return times;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> cpu_mttkrp_iegenlib(Config config, char *filename) {
  if (config.debug) {
    std::cout << "cpu mttkrp iegenlib =====\n";
  }

  DataForCpuMttkrp data(filename, config);

#define A(i, j) data.a.data[i * J + j]
#define B(z) data.bValues.data[z]
#define D(l, j) data.d.data[l * J + j]
#define C(k, j) data.c.data[k * J + j]

  uint64_t *UFi = data.bCoord0.data;
  uint64_t *UFk = data.bCoord1.data;
  uint64_t *UFl = data.bCoord2.data;

  uint64_t NNZ = data.NNZ;
  uint64_t J = data.J;

  uint64_t t1, t2, t3, t4, t5;

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    auto start = milliTime();
    // Generated code ==============================

#undef s0
#undef s_0
#define s_0(z, i, k, l, j) A(i, j) += B(z) * D(l, j) * C(k, j)
#define s0(z, i, k, l, j) s_0(z, i, k, l, j);

#undef UFi_0
#undef UFk_1
#undef UFl_2
#define UFi(t0) UFi[t0]
#define UFi_0(__tv0) UFi(__tv0)
#define UFk(t0) UFk[t0]
#define UFk_1(__tv0) UFk(__tv0)
#define UFl(t0) UFl[t0]
#define UFl_2(__tv0) UFl(__tv0)

    t1 = 0;
    t2 = 0;
    t3 = 0;
    t4 = 0;
    t5 = 0;

    if (J >= 1) {
      for (t1 = 0; t1 <= NNZ - 1; t1++) {
        t2 = UFi_0(t1);
        t3 = UFk_1(t1);
        t4 = UFl_2(t1);
        for (t5 = 0; t5 <= J - 1; t5++) {
          s0(t1, t2, t3, t4, t5);
        }
      }
    }

#undef s0
#undef s_0
#undef UFi_0
#undef UFk_1
#undef UFl_2

    // =============================================
    auto stop = milliTime();
    times[i] = stop - start;
  }

  if (config.debug) {
    data.dump();
    std::cout << "=====\n";
  }

  if (config.test) {
    runTest(data, "cpu mttkrp iegenlib", filename, config);
  }

  return times;
}

std::vector<int64_t> cpu_ttm_mlir(Config config, char *filename) {
  if (config.debug) {
    std::cout << "cpu ttm mlir =====\n";
  }

  DataForCpuTTM data(filename, config);

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    times[i] = _mlir_ciface_sparse_ttm_cpu(data.Mf, data.R, &data.fptr,
                                           &data.xCoordConstant, &data.xValues,
                                           &data.u, &data.y);
  }

  if (config.debug) {
    data.dump();
    std::cout << "=====\n";
  }

  if (config.test) {
    runTest(data, "cpu ttm mlir", filename, config);
  }

  return times;
}

std::vector<int64_t> cpu_ttm_iegenlib(Config config, char *filename) {
  if (config.debug) {
    std::cout << "cpu ttm iegenlib =====\n";
  }

  auto data = DataForCpuTTM(filename, config);

  uint64_t R = data.R;
  uint64_t Mf = data.Mf;

  uint64_t *UFfptr = data.fptr.data;
  uint64_t *UFr = data.xCoordConstant.data;

#define Y(i, k) data.y.data[i * R + k]
#define X(z) data.xValues.data[z]
#define U(r, k) data.u.data[r * R + k]

  uint64_t t1, t2, t3, t4;

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    auto start = milliTime();
    // Generated code ==============================

#undef s0
#undef s_0
#define s_0(z, j, r, k) Y(z, k) += X(j) * U(r, k)
#define s0(z, j, r, k) s_0(z, j, r, k);

#undef UFfptr_1
#undef UFfptr_2
#undef UFr_0
#define UFfptr(t0) UFfptr[t0]
#define UFfptr_1(__tv0) UFfptr(__tv0)
#define UFfptr_2(__tv0) UFfptr(__tv0 + 1)
#define UFr(t0) UFr[t0]
#define UFr_0(__tv0, __tv1) UFr(__tv1)

    t1 = 0;
    t2 = 0;
    t3 = 0;
    t4 = 0;

    if (R >= 1) {
      for (t1 = 0; t1 <= Mf - 1; t1++) {
        for (t2 = UFfptr_1(t1); t2 <= UFfptr_2(t1) - 1; t2++) {
          t3 = UFr_0(t1, t2);
          for (t4 = 0; t4 <= R - 1; t4++) {
            s0(t1, t2, t3, t4);
          }
        }
      }
    }

#undef s0
#undef s_0
#undef UFfptr_1
#undef UFfptr_2
#undef UFr_0

    // =============================================
    auto stop = milliTime();
    times[i] = stop - start;
  }

  if (config.debug) {
    data.dump();
    std::cout << "=====\n";
  }

  if (config.test) {
    runTest(data, "cpu ttm iegenlib", filename, config);
  }

  return times;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> gpu_mttkrp_mlir(Config config, char *filename) {
  if (config.debug) {
    std::cout << "gpu mttkrp mlir =====\n";
  }

  DataForGpuMttkrp data(filename, config);

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    times[i] = _mlir_ciface_sparse_mttkrp_gpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (config.debug || config.test) {
    auto cpuData = DataForCpuMttkrp(data);

    if (config.debug) {
      cpuData.dump();
      std::cout << "=====\n";
    }

    if (config.test) {
      runTest(cpuData, "gpu mttkrp mlir", filename, config);
    }
  }

  return times;
}

std::vector<int64_t> gpu_ttm_mlir(Config config, char *filename) {
  if (config.debug) {
    std::cout << "gpu mttkrp mlir =====\n";
  }

  DataForGpuTTM data(filename, config);

  std::vector<int64_t> times(config.iterations);
  for (int64_t i = 0; i < config.iterations; i++) {
    times[i] = _mlir_ciface_sparse_ttm_gpu(data.Mf, data.R, &data.fptr,
                                           &data.xCoordConstant, &data.xValues,
                                           &data.u, &data.y);
  }

  if (config.debug || config.test) {
    auto cpuData = DataForCpuTTM(data);

    if (config.debug) {
      cpuData.dump();
      std::cout << "=====\n";
    }

    if (config.test) {
      runTest(cpuData, "gpu ttm mlir", filename, config);
    }
  }

  return times;
}
