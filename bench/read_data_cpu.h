#ifndef READ_DATA_CPU_H
#define READ_DATA_CPU_H

#include "Runtime/CPURuntime.h"
#include "Runtime/GPURuntime.h"
#include "benchmarks.h"
#include "read_data_gpu.h"

// variable names from http://tensor-compiler.org/docs/data_analytics
class DataForCpuMttkrp {
  COO *bData;
  std::vector<double> cData;
  std::vector<double> dData;
  std::vector<double> aData;

  void runReferenceImplementation() {
    for (uint64_t x = 0; x < NNZ; x++) {
      uint64_t i = bCoord0.data[x];
      uint64_t k = bCoord1.data[x];
      uint64_t l = bCoord2.data[x];
      for (uint64_t j = 0; j < J; j++) {
        a.data[i * J + j] +=
            bValues.data[x] * d.data[l * J + j] * c.data[k * J + j];
      }
    }
  }

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

  DataForCpuMttkrp(char *filename, Config config, bool isReference = false)
      : bData((COO *)_mlir_ciface_read_coo(filename)), NNZ(bData->nnz),
        I(bData->dims[0]), J(config.J), K(bData->dims[1]), L(bData->dims[2]) {

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

    if (isReference) {
      runReferenceImplementation();
    }
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

  bool isSame(DataForCpuMttkrp &other) {
    if (this->NNZ != other.NNZ) {
      return false;
    }
    if (this->I != other.I) {
      return false;
    }
    if (this->J != other.J) {
      return false;
    }
    if (this->K != other.K) {
      return false;
    }
    if (this->L != other.L) {
      return false;
    }

    // coords
    if (this->bData->coord[0] != other.bData->coord[0]) {
      return false;
    };
    if (this->bData->coord[1] != other.bData->coord[1]) {
      return false;
    };
    if (this->bData->coord[2] != other.bData->coord[2]) {
      return false;
    };
    if (this->bData->values != other.bData->values) {
      return false;
    };

    // inputs
    if (this->cData != other.cData) {
      return false;
    };
    if (this->dData != other.dData) {
      return false;
    };

    // output
    if (this->aData != other.aData) {
      return false;
    };

    return true;
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

  DataForCpuTTM(char *filename, Config config)
      : xData((COO *)_mlir_ciface_read_coo(filename)), I(xData->dims[0]),
        J(xData->dims[1]), K(xData->dims[2]), R(config.R) {
    assert(xData->rank == 3 && "ttm only supports rank 3 tensor");
    assert(config.constantMode < 3 && "constant mode dimension not in bounds");

    uint64_t constantModeDimSize = xData->dims[config.constantMode];

    // Construct memrefs pointing into xData
    _mlir_ciface_coords(&xCoordConstant, xData, config.constantMode);
    _mlir_ciface_values(&xValues, xData);

    // Sort data lexigraphically with constant mode considered the last. We
    // need to do this to be able to calculate fibers.
    xData->sortIndicesModeLast(config.constantMode);

    // construct data for fptr
    fptrData = fiberStartStopIndices(*xData, config.constantMode);
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
    //      0  0  0  1  1  1  1  2  2  2  2
    //      sptIndexVector length: 11
    //      0  2  3  0  1  2  3  0  1  2  3
    //      values: 11 x 2 matrix 154.00  231.00 20.00   33.00
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

#endif // READ_DATA_CPU_H