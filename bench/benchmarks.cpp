#include <cstdint>
#include <functional>
#include <iostream>
#include <ratio>
#include <vector>

#include "Runtime/CPURuntime.h"
#include "Runtime/GPURuntime.h"
#include "benchmarks.h"

extern "C" {
int64_t _mlir_ciface_sparse_ttm_cpu(
    uint64_t Mf, uint64_t R, StridedMemRefType<uint64_t, 1> *fptr,
    StridedMemRefType<uint64_t, 1> *x_coord_constant,
    StridedMemRefType<double, 1> *x_values, StridedMemRefType<double, 2> *u,
    StridedMemRefType<double, 2> *y);
int64_t _mlir_ciface_sparse_mttkrp_cpu(
    uint64_t NNZ, uint64_t J, StridedMemRefType<uint64_t, 1> *b_coord_0,
    StridedMemRefType<uint64_t, 1> *b_coord_1,
    StridedMemRefType<uint64_t, 1> *b_coord_2,
    StridedMemRefType<double, 1> *b_values, StridedMemRefType<double, 2> *c,
    StridedMemRefType<double, 2> *d, StridedMemRefType<double, 2> *a);
int64_t _mlir_ciface_sparse_ttm_gpu(
    uint64_t Mf, uint64_t R, StridedMemRefType<uint64_t, 1> *fptr,
    StridedMemRefType<uint64_t, 1> *x_coord_constant,
    StridedMemRefType<double, 1> *x_values, StridedMemRefType<double, 2> *u,
    StridedMemRefType<double, 2> *y);
int64_t _mlir_ciface_sparse_mttkrp_gpu(
    uint64_t NNZ, uint64_t J, StridedMemRefType<uint64_t, 1> *b_coord_0,
    StridedMemRefType<uint64_t, 1> *b_coord_1,
    StridedMemRefType<uint64_t, 1> *b_coord_2,
    StridedMemRefType<double, 1> *b_values, StridedMemRefType<double, 2> *c,
    StridedMemRefType<double, 2> *d, StridedMemRefType<double, 2> *a);
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

    // Construct memrefs pointing into bData
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

    bData = new COO(NNZ, 3, {I, K, L}, std::move(coord), std::move(values));

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

    // Construct memrefs pointing into xData
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

  // This constructor expects src DataForGpuTTM to be populated
  DataForCpuTTM(DataForGpuTTM &src)
      : Mf(src.Mf), I(src.I), J(src.J), K(src.K), R(src.R) {

    std::vector<std::vector<uint64_t>> coord;
    coord.push_back(copyToCpuMemRef(&src.xCoordConstant, &xCoordConstant));

    std::vector<double> values = copyToCpuMemRef(&src.xValues, &xValues);

    xData = new COO(values.size(), 1,
                    {static_cast<unsigned long>(src.xCoordConstant.sizes[0])},
                    std::move(coord), std::move(values));

    fptrData = copyToCpuMemRef(&src.fptr, &fptr);
    uData = copyToCpuMemRef(&src.u, &u);
    yData = copyToCpuMemRef(&src.y, &y);
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

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> cpu_mttkrp_mlir(bool debug, int64_t iterations,
                                     char *filename) {
  if (debug) {
    std::cout << "cpu mttkrp mlir =====\n";
  }

  DataForCpuMttkrp data(filename, 650);

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
    times[i] = _mlir_ciface_sparse_mttkrp_cpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return times;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> cpu_mttkrp_iegenlib(bool debug, int64_t iterations,
                                         char *filename) {
  if (debug) {
    std::cout << "cpu mttkrp iegenlib =====\n";
  }

  DataForCpuMttkrp data(filename, 650);

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

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
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

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return times;
}

std::vector<int64_t> cpu_ttm_mlir(bool debug, int64_t iterations,
                                  char *filename) {
  if (debug) {
    std::cout << "cpu ttm mlir =====\n";
  }

  DataForCpuTTM data(filename, 0, 2);

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
    times[i] = _mlir_ciface_sparse_ttm_cpu(data.Mf, data.R, &data.fptr,
                                           &data.xCoordConstant, &data.xValues,
                                           &data.u, &data.y);
  }

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return times;
}

std::vector<int64_t> cpu_ttm_iegenlib(bool debug, int64_t iterations,
                                      char *filename) {
  auto data = DataForCpuTTM(filename, 0, 2);

  uint64_t R = data.R;
  uint64_t Mf = data.Mf;

  uint64_t *UFfptr = data.fptr.data;
  uint64_t *UFr = data.xCoordConstant.data;

#define Y(i, k) data.y.data[i * R + k]
#define X(z) data.xValues.data[z]
#define U(r, k) data.u.data[r * R + k]

  uint64_t t1, t2, t3, t4;

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
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

  if (debug) {
    data.dump();
    std::cout << "=====\n";
  }

  return times;
}

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
std::vector<int64_t> gpu_mttkrp_mlir(bool debug, int64_t iterations,
                                     char *filename) {
  if (debug) {
    std::cout << "gpu mttkrp mlir =====\n";
  }

  DataForGpuMttkrp data(filename, 650);

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
    times[i] = _mlir_ciface_sparse_mttkrp_gpu(
        data.NNZ, data.J, &data.bCoord0, &data.bCoord1, &data.bCoord2,
        &data.bValues, &data.c, &data.d, &data.a);
  }

  if (debug) {
    DataForCpuMttkrp(data).dump();
    std::cout << "=====\n";
  }

  return times;
}

std::vector<int64_t> gpu_ttm_mlir(bool debug, int64_t iterations,
                                  char *filename) {
  if (debug) {
    std::cout << "gpu mttkrp mlir =====\n";
  }

  DataForGpuTTM data(filename, 0, 2);

  std::vector<int64_t> times(iterations);
  for (int64_t i = 0; i < iterations; i++) {
    times[i] = _mlir_ciface_sparse_ttm_gpu(data.Mf, data.R, &data.fptr,
                                           &data.xCoordConstant, &data.xValues,
                                           &data.u, &data.y);
  }

  if (debug) {
    DataForCpuTTM(data).dump();
    std::cout << "=====\n";
  }

  return times;
}
