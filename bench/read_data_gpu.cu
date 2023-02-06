#include <functional>

#include "Runtime/GPURuntime.h"
#include "read_data_gpu.h"

void allocateAndPopulateGpuMemref(
    StridedMemRefType<float, 2> *ref, uint64_t dim1, uint64_t dim2,
    std::function<float(uint64_t i, uint64_t j)> &&fill) {
  float *d_data;
  uint64_t size = dim1 * dim2;

  std::vector<float> data = std::vector<float>(size);
  for (uint64_t i = 0; i < dim1; i++) {
    for (uint64_t j = 0; j < dim2; j++) {
      data[i * dim2 + j] = fill(i, j);
    }
  }

  cudaMalloc(&d_data, sizeof(float) * size);
  cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

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

DataForGpuMttkrp::DataForGpuMttkrp(char *filename, Config config)
    : bData((COO *)_mlir_ciface_read_coo(filename)), NNZ(bData->nnz),
      I(bData->dims[0]), J(config.J), K(bData->dims[1]), L(bData->dims[2]) {
  assert(bData->rank == 3 && "mttkrp requires rank 3 tensor");

  // read data from b into gpu memory and construct memref into said data
  _mlir_ciface_coords_gpu(&bCoord0, bData, 0);
  _mlir_ciface_coords_gpu(&bCoord1, bData, 1);
  _mlir_ciface_coords_gpu(&bCoord2, bData, 2);
  _mlir_ciface_values_gpu(&bValues, bData);

  // Construct c matrix (K x J)
  allocateAndPopulateGpuMemref(
      &c, K, J, [](uint64_t i, uint64_t j) -> float { return 1.0; });

  // Construct d matrix (L x J)
  allocateAndPopulateGpuMemref(
      &d, L, J, [](uint64_t i, uint64_t j) -> float { return 1.0; });

  // Construct a matrix (I x J)
  allocateAndPopulateGpuMemref(
      &a, I, J, [](uint64_t i, uint64_t j) -> float { return 0.0; });
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

DataForGpuTTM::DataForGpuTTM(char *filename, Config config)
    : xData((COO *)_mlir_ciface_read_coo(filename)), I(xData->dims[0]),
      J(xData->dims[1]), K(xData->dims[2]), R(config.R),
      constantMode(config.constantMode) {
  assert(xData->rank == 3 && "ttm only supports rank 3 tensor");
  assert(config.constantMode < 3 && "constant mode dimension not in bounds");

  uint64_t constantModeDimSize = xData->dims[config.constantMode];

  // Sort data lexicographically with constant mode considered the last. We need
  // to do this to be able to calculate fibers.
  xData->sortIndicesModeLast(config.constantMode);

  // read data from x into gpu memory and construct memref into said data
  _mlir_ciface_coords_gpu(&xCoordConstant, xData, config.constantMode);
  _mlir_ciface_values_gpu(&xValues, xData);

  // construct fptr vector
  auto fptrData = fiberStartStopIndices(*xData, config.constantMode);
  allocateAndPopulateGpuMemref(
      &fptr, fptrData.size(),
      [&](uint64_t i) -> uint64_t { return fptrData[i]; });

  Mf = fptrData.size() - 1;

  // construct u matrix (constantModeDimSize x R)
  allocateAndPopulateGpuMemref(
      &u, constantModeDimSize, R,
      [=](uint64_t i, uint64_t j) -> float { return 1.0; });

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
      &y, Mf, R, [](uint64_t i, uint64_t j) -> float { return 0.0; });
}

DataForGpuTTM::~DataForGpuTTM() {
  delete xData;
  cudaFree(fptr.data);
  cudaFree(xCoordConstant.data);
  cudaFree(xValues.data);
  cudaFree(u.data);
  cudaFree(y.data);
}