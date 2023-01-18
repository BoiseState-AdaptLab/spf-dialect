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

// variable names from http://tensor-compiler.org/docs/data_analytics
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
std::vector<uint64_t> fiberStartStopIndices(COO &sortedCoo,
                                            uint64_t constantMode) {
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

// variable names from PASTA paper: https://arxiv.org/abs/1902.03317
class DataForCpuTTM {
  COO *xData;
  std::vector<uint64_t> fptrData;
  std::vector<double> uData;
  std::vector<double> yData;

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

  DataForCpuTTM(char *filename, uint64_t constantMode, uint64_t inR)
      : xData((COO *)_mlir_ciface_read_coo(filename)), I(xData->dims[0]),
        J(xData->dims[1]), K(xData->dims[2]), R(inR) {
    assert(xData->rank == 3 && "ttm only supports rank 3 tensor");
    assert(constantMode < 3 && "constant mode dimension not in bounds");

    uint64_t constantModeDimSize = xData->dims[constantMode];

    // read data from x into memrefs
    _mlir_ciface_coords(&xCoordConstant, xData, constantMode);
    _mlir_ciface_values(&xValues, xData);

    // Sort data lexigraphically with constant mode considered the last. We need
    // to do this to be able to calculate fibers.
    xData->sortIndicesModeLast(constantMode);

    // construct data for fptr
    fptrData = fiberStartStopIndices(*xData, constantMode);
    Mf = fptrData.size() - 1;
    fptr.basePtr = fptr.data = fptrData.data();
    fptr.offset = 0;
    fptr.sizes[0] = fptrData.size();
    fptr.strides[0] = 1;

    // construct data for u matrix
    uData = std::vector<double>(constantModeDimSize * R);
    for (uint64_t i = 0; i < constantModeDimSize; i++) {
      for (uint64_t j = 0; j < R; j++) {
        uData[i * R + j] = i * R + j;
      }
    }
    u.basePtr = u.data = uData.data();
    u.offset = 0;
    u.sizes[0] = constantModeDimSize;
    u.sizes[1] = R;
    u.strides[0] = R;
    u.strides[1] = 1;

    // construct data for y matrix
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
    yData = std::vector<double>(Mf * R);
    std::fill(yData.begin(), yData.end(), 0.0);
    y.basePtr = y.data = yData.data();
    y.offset = 0;
    y.sizes[0] = Mf;
    y.sizes[1] = R;
    y.strides[0] = R;
    y.strides[1] = 1;
  }

  ~DataForCpuTTM() { delete xData; }

  void dump() {
    std::cout << "I: " << this->I << "\n";
    std::cout << "J: " << this->J << "\n";
    std::cout << "K: " << this->K << "\n";
    std::cout << "R: " << this->R << "\n";
    std::cout << "Mf: " << this->Mf << "\n";
    std::cout << "xCoordConstant:\n";
    impl::printMemRef(this->xCoordConstant);
    std::cout << "xValues:\n";
    impl::printMemRef(this->xValues);
    std::cout << "fptr:\n";
    impl::printMemRef(this->fptr);
    std::cout << "u:\n";
    impl::printMemRef(this->u);
    std::cout << "y:\n";
    impl::printMemRef(this->y);
  }
};
} // anonymous namespace

int64_t cpu_ttm_iegenlib(bool debug, int64_t iterations, char *filename) {
  auto data = DataForCpuTTM(filename, 0, 2);

  uint64_t R = data.R;
  uint64_t Mf = data.Mf;

  uint64_t *UFfptr = data.fptr.data;
  uint64_t *UFr = data.xCoordConstant.data;

#define Y(i, k) data.y.data[i * R + k]
#define X(z) data.xValues.data[z]
#define U(r, k) data.u.data[r * R + k]

  uint64_t t1, t2, t3, t4, t5, t6;

  int64_t totalTime = 0;
  auto start = milliTime();
  for (int64_t i = 0; i < iterations; i++) {
    // Generated code ==============================

#undef s0
#undef s_0
#define s_0(z, ib, ie, j, r, k) Y(z, k) += X(j) * U(r, k)
#define s0(z, ib, ie, j, r, k) s_0(z, ib, ie, j, r, k);

#undef UFfptr_0
#undef UFfptr_1
#undef UFr_2
#define UFfptr(t0) UFfptr[t0]
#define UFfptr_0(__tv0) UFfptr(__tv0)
#define UFfptr_1(__tv0) UFfptr(__tv0 + 1)
#define UFr(t0) UFr[t0]
#define UFr_2(__tv0, __tv1, __tv2, __tv3) UFr(__tv3)

    t1 = 0;
    t2 = 0;
    t3 = 0;
    t4 = 0;
    t5 = 0;
    t6 = 0;

    if (R >= 1) {
      for (t1 = 0; t1 <= Mf - 1; t1++) {
        t2 = UFfptr_0(t1);
        t3 = UFfptr_1(t1);
        for (t4 = UFfptr_0(t1); t4 <= UFfptr_1(t1) - 1; t4++) {
          t5 = UFr_2(t1, t2, t3, t4);
          for (t6 = 0; t6 <= R - 1; t6++) {
            s0(t1, t2, t3, t4, t5, t6);
          }
        }
      }
    }

#undef s0
#undef s_0
#undef UFfptr_0
#undef UFfptr_1
#undef UFr_2

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