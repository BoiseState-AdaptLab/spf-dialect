#include <cstdint>
#include <functional>
#include <iostream>
#include <ratio>
#include <vector>

#include "Runtime/CPURuntime.h"
#include "Runtime/GPURuntime.h"
#include "benchmarks.h"

extern "C" {
int64_t _mlir_ciface_sparse_mttkrp_cpu(uint64_t NNZ, uint64_t J,
                                       StridedMemRefType<uint64_t, 1> *coord_0,
                                       StridedMemRefType<uint64_t, 1> *coord_1,
                                       StridedMemRefType<uint64_t, 1> *coord_2,
                                       StridedMemRefType<double, 1> *values,
                                       StridedMemRefType<double, 2> *c,
                                       StridedMemRefType<double, 2> *d,
                                       StridedMemRefType<double, 2> *a);
int64_t _mlir_ciface_sparse_mttkrp_gpu(uint64_t NNZ, uint64_t J,
                                       StridedMemRefType<uint64_t, 1> *coord_0,
                                       StridedMemRefType<uint64_t, 1> *coord_1,
                                       StridedMemRefType<uint64_t, 1> *coord_2,
                                       StridedMemRefType<double, 1> *values,
                                       StridedMemRefType<double, 2> *c,
                                       StridedMemRefType<double, 2> *d,
                                       StridedMemRefType<double, 2> *a);
}

namespace {
class DataForCpuMttkrp {
  COO *bData;
  std::vector<double> cData;
  std::vector<double> dData;
  std::vector<double> aData;

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

  DataForCpuMttkrp(char *filename, uint64_t inJ)
      : bData((COO *)_mlir_ciface_read_coo(filename)), NNZ(bData->nnz),
        I(bData->dims[0]), J(inJ), K(bData->dims[1]), L(bData->dims[2]) {

    assert(bData->rank == 3 && "mttkrp requires rank 3 tensor");
    _mlir_ciface_coords(&bCoord0, bData, 0);
    _mlir_ciface_coords(&bCoord1, bData, 1);
    _mlir_ciface_coords(&bCoord2, bData, 2);
    _mlir_ciface_values(&bValues, bData);

    // Construct c matrix
    cData = std::vector<double>(K * J);
    for (uint64_t k = 0; k < K; k++) {
      for (uint64_t j = 0; j < J; j++) {
        cData[k * J + j] = k * J + j;
      }
    }
    c.basePtr = c.data = cData.data();
    c.offset = 0;
    c.sizes[0] = K;
    c.sizes[1] = J;
    c.strides[0] = J;
    c.strides[1] = 1;

    // Construct d matrix
    dData = std::vector<double>(L * J);
    for (uint64_t l = 0; l < L; l++) {
      for (uint64_t j = 0; j < J; j++) {
        dData[l * J + j] = l * J + j;
      }
    }
    d.basePtr = d.data = dData.data();
    d.offset = 0;
    d.sizes[0] = L;
    d.sizes[1] = J;
    d.strides[0] = J;
    d.strides[1] = 1;

    // Construct a matrix
    aData = std::vector<double>(I * J);
    std::fill(aData.begin(), aData.end(), 0.0);
    a.basePtr = a.data = aData.data();
    a.offset = 0;
    a.sizes[0] = I;
    a.sizes[1] = J;
    a.strides[0] = J;
    a.strides[1] = 1;
  }

  // This constructor expects src DataForGpuMttkrp to be populated
  DataForCpuMttkrp(DataForGpuMttkrp &src)
      : NNZ(src.NNZ), I(src.I), J(src.J), K(src.K), L(src.L) {

    std::vector<std::vector<uint64_t>> coord;
    coord.push_back(copyToCpuMemRef(&src.bCoord0, &bCoord0));
    coord.push_back(copyToCpuMemRef(&src.bCoord1, &bCoord1));
    coord.push_back(copyToCpuMemRef(&src.bCoord2, &bCoord2));

    std::vector<double> values = copyToCpuMemRef(&src.bValues, &bValues);

    bData = new COO(NNZ, 3, {I, J, K, L}, std::move(coord), std::move(values));

    cData = copyToCpuMemRef(&src.c, &c);
    dData = copyToCpuMemRef(&src.d, &d);
    aData = copyToCpuMemRef(&src.a, &a);
  }
  ~DataForCpuMttkrp() { delete bData; }

  void dump() {
    std::cout << "NNZ: " << this->NNZ << "\n";
    std::cout << "I: " << this->I << "\n";
    std::cout << "J: " << this->J << "\n";
    std::cout << "K: " << this->K << "\n";
    std::cout << "L: " << this->L << "\n";
    std::cout << "bCoord0:\n";
    impl::printMemRef(this->bCoord0);
    std::cout << "bCoord1:\n";
    impl::printMemRef(this->bCoord1);
    std::cout << "bCoord2:\n";
    impl::printMemRef(this->bCoord2);
    std::cout << "bValues:\n";
    impl::printMemRef(this->bValues);
    std::cout << "c:\n";
    impl::printMemRef(this->c);
    std::cout << "d:\n";
    impl::printMemRef(this->d);
    std::cout << "a:\n";
    impl::printMemRef(this->a);
  }
};
} // anonymous namespace

// areCoordsEqualExceptMode returns true if coords are equal in all modes
// <exceptMode>
bool areCoordsEqualExceptMode(COO &coo, uint64_t exceptMode, uint64_t i,
                              uint64_t j) {
  for (uint64_t mode = 0; mode < coo.rank; mode++) {
    if (mode != exceptMode) {
      auto one = coo.coord[mode][i];
      auto two = coo.coord[mode][j];
      if (one != two) {
        return false;
      }
    }
  }
  return true;
}

// fiberStartStopIndices returns the indices at which fibers in COO tensor
// <sortedCoo> formed by holding <constantMode> constant begin and end.
// <sortedCoo> is assumed to have been sorted lexigraphically with
// <constantMode> considered that last mode.
//
// Ex: the following COO mode 3 tensor is sorted lexigraphically with mode 0
//    (constant mode) last
// mode: 0,1,2
// 0|    1 0 0 : 77
// 1|    0 0 2 : 3
// 2|    1 0 2 : 10
// 3|    0 0 3 : 63
// ⬑ index
// output will be: [0,1,3,4].
//  - Fiber 0 starts at index 0 (always)
//  - Fiber 0 ends (and fiber 1 starts) at index 1 (as at index 1, one two one
//    of the non-constant dimensions from index 0 differ at mode 2: 0 to 2)
//  - Fiber 1 (and fiber 2 starts) ends at index 3 (as at index 3 one of the two
//    non-constant dimensions that are the same for index 1 and 2 differ at mode
//    2: 2 to 3)
//  - As fiber 2 is the last fiber it ends at the last index +1;
std::vector<uint64_t> fiberStartStopIndices(COO &sortedCoo, uint64_t constantMode) {
  std::vector<uint64_t> out;
  uint64_t lastIdx = sortedCoo.nnz;
  for (uint64_t i = 0; i < sortedCoo.nnz; i++) {
    if (lastIdx == sortedCoo.nnz ||
        !areCoordsEqualExceptMode(sortedCoo, constantMode, lastIdx, i)) {
      lastIdx = i;
      out.push_back(i);
    }
  }
  out.push_back(sortedCoo.nnz);
  return out;
}

int64_t cpu_ttm_iegenlib(bool debug, int64_t iterations, char *filename) {
  COO *Xdata = (COO *)_mlir_ciface_read_coo(filename);

  uint64_t constantMode = 0;

  Xdata->dump(std::cout);
  Xdata->sortIndicesModeLast(0);
  std::cout << "============================================\n";
  Xdata->dump(std::cout);

  std::vector<uint64_t> fiberStartStop = fiberStartStopIndices(*Xdata, constantMode);

  bool first = true;
  std::cout << "fg: [";
  for (auto i : fiberStartStop) {
    if (first) {
      first = false;
    } else {
      std::cout << ", ";
    }
    std::cout << i;
  }
  std::cout << "]\n";

  uint64_t NFIBERS = fiberStartStop.size() -1;

  uint64_t NCOLS = 2;

    // Construct U matrix
    StridedMemRefType<double, 2> u;
    auto un = Xdata->dims[constantMode];
    auto um = NCOLS;
    auto uData = std::vector<double>(un * um);
    for (uint64_t i = 0; i < un; i++) {
      for (uint64_t j = 0; j < um; j++) {
        uData[i * um + j] = i * um + j;
      }
    }
    u.basePtr = u.data = uData.data();
    u.offset = 0;
    u.sizes[0] = un;
    u.sizes[1] = um;
    u.strides[0] = um;
    u.strides[1] = 1;

    // Construct Y matrix
    StridedMemRefType<double, 2> y;
    auto yn = NFIBERS;
    auto ym = NCOLS;
    auto yData = std::vector<double>(NFIBERS * NCOLS);
    std::fill(yData.begin(), yData.end(), 0.0);
    y.basePtr = y.data = yData.data();
    y.offset = 0;
    y.sizes[0] = yn;
    y.sizes[1] = ym;
    y.strides[0] = ym;
    y.strides[1] = 1;

#define Y(i, k) y.data[i * NCOLS + k]
#define X(z) Xdata->values[z]
#define U(r, k) u.data[r * NCOLS + k]

  uint64_t *UFib = fiberStartStop.data();

  uint64_t *UFr = Xdata->coord[constantMode].data();

  uint64_t t1, t2, t3, t4, t5, t6;

  // Generated code ==============================

#undef s0
#undef s_0
#define s_0(z, ib, ie, j, r, k)   Y(z,k) += X(j) * U(r,k)
#define s0(z, ib, ie, j, r, k)   s_0(z, ib, ie, j, r, k);

#undef UFib_0
#undef UFib_1
#undef UFr_2
#define UFib(t0) UFib[t0]
#define UFib_0(__tv0) UFib(__tv0)
#define UFib_1(__tv0) UFib(__tv0 + 1)
#define UFr(t0) UFr[t0]
#define UFr_2(__tv0, __tv1, __tv2, __tv3) UFr(__tv3)

t1 = 0;
t2 = 0;
t3 = 0;
t4 = 0;
t5 = 0;
t6 = 0;

if (NCOLS >= 1) {
  for(t1 = 0; t1 <= NFIBERS-1; t1++) {
    t2=UFib_0(t1);
    t3=UFib_1(t1);
    for(t4 = UFib_0(t1); t4 <= UFib_1(t1)-1; t4++) {
      t5=UFr_2(t1,t2,t3,t4);
      for(t6 = 0; t6 <= NCOLS-1; t6++) {
        s0(t1,t2,t3,t4,t5,t6);
      }
    }
  }
}

#undef s0
#undef s_0
#undef UFib_0
#undef UFib_1
#undef UFr_2

  // =============================================

  first = true;
  std::cout << "y: [";
  for (auto i : y) {
    if (first) {
      first = false;
    } else {
      std::cout << ", ";
    }
    std::cout << i;
  }
  std::cout << "]\n";
  return 0;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
int64_t cpu_mttkrp_iegenlib(bool debug, int64_t iterations, char *filename) {
  if (debug) {
    std::cout << "cpu mttkrp iegenlib =====\n";
  }

  DataForCpuMttkrp data(filename, 5);

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

  int64_t totalTime = 0;
  auto start = milliTime();
  for (int64_t i = 0; i < iterations; i++) {
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
    totalTime += stop - start;
  }

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return totalTime / iterations;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
int64_t gpu_mttkrp_mlir(bool debug, int64_t iterations, char *filename) {
  if (debug) {
    std::cout << "gpu mttkrp mlir =====\n";
  }

  DataForGpuMttkrp data(filename, 5);

  int64_t totalTime = 0;
  for (int64_t i = 0; i < iterations; i++) {
    totalTime += _mlir_ciface_sparse_mttkrp_gpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (debug) {
    DataForCpuMttkrp(data).dump();
    std::cout << "=====\n";
  }

  return totalTime / iterations;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
int64_t cpu_mttkrp_mlir(bool debug, int64_t iterations, char *filename) {
  if (debug) {
    std::cout << "cpu mttkrp mlir =====\n";
  }

  DataForCpuMttkrp data(filename, 5);

  int64_t totalTime = 0;
  for (int64_t i = 0; i < iterations; i++) {
    totalTime += _mlir_ciface_sparse_mttkrp_cpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return totalTime / iterations;
}