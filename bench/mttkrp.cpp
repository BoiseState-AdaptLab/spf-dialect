#include "Runtime/runtime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" {
void sparse_mttkrp(uint64_t nnz, uint64_t I, uint64_t J, uint64_t K, uint64_t L,
                   StridedMemRefType<uint64_t, 1> coord_0,
                   StridedMemRefType<uint64_t, 1> coord_1,
                   StridedMemRefType<uint64_t, 1> coord_2,
                   StridedMemRefType<double, 1> values,
                   StridedMemRefType<double, 2> c,
                   StridedMemRefType<double, 2> d,
                   StridedMemRefType<double, 2> a);
}

int main() {
  printf("EXPECTED OUTPUT =========================\n");
  printf("Unranked Memref base@ = 0x558bc8cb08e0 rank = 2 offset = 0 sizes = "
         "[3, 5] strides = [5, 1] data = \n");
  printf("[[0,   1,   2,   3,   4], \n");
  printf(" [5,   6,   7,   8,   9], \n");
  printf(" [10,   11,   12,   13,   14]]\n");
  printf("Unranked Memref base@ = 0x558bc8cd20a0 rank = 2 offset = 0 sizes = "
         "[4, 5] strides = [5, 1] data = \n");
  printf("[[0,   1,   2,   3,   4], \n");
  printf(" [5,   6,   7,   8,   9], \n");
  printf(" [10,   11,   12,   13,   14], \n");
  printf(" [15,   16,   17,   18,   19]]\n");
  printf("Unranked Memref base@ = 0x558bc8c26cd0 rank = 2 offset = 0 sizes = "
         "[2, 5] strides = [5, 1] data = \n");
  printf("[[0,   0,   0,   0,   0], \n");
  printf(" [0,   0,   0,   0,   0]]\n");
  printf("Unranked Memref base@ = 0x558bc8c26cd0 rank = 2 offset = 0 sizes = "
         "[2, 5] strides = [5, 1] data = \n");
  printf("[[16075,   21930,   28505,   35800,   43815], \n");
  printf(" [10000,   14225,   19180,   24865,   31280]]\n");
  printf("OUTPUT ==================================\n");

  char *filename = getTensorFilename(0);

  COO *coo = (COO *)_mlir_ciface_read_coo(filename);

  assert(coo->rank == 3 && "mttkrp requires rank 3 tensor");

  StridedMemRefType<uint64_t, 1> bCoord0;
  _mlir_ciface_coords(&bCoord0, coo, 0);
  StridedMemRefType<uint64_t, 1> bCoord1;
  _mlir_ciface_coords(&bCoord1, coo, 1);
  StridedMemRefType<uint64_t, 1> bCoord2;
  _mlir_ciface_coords(&bCoord2, coo, 2);
  StridedMemRefType<double, 1> bVals;
  _mlir_ciface_values(&bVals, coo);

  uint64_t I = coo->dims[0];
  uint64_t J = 5;
  uint64_t K = coo->dims[1];
  uint64_t L = coo->dims[2];
  uint64_t nnz = coo->nnz;

  std::vector<double> cd = std::vector<double>(K * J);
  for (uint64_t k = 0; k < K; k++) {
    for (uint64_t j = 0; j < J; j++) {
      cd[k * J + j] = k * J + j;
    }
  }
  StridedMemRefType<double, 2> c;
  c.basePtr = c.data = cd.data();
  c.offset = 0;
  c.sizes[0] = K;
  c.sizes[1] = J;
  c.strides[0] = J;
  c.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&c};
    _mlir_ciface_printMemrefF64(&unranked);
  }

  std::vector<double> dc = std::vector<double>(L * J);
  for (uint64_t l = 0; l < L; l++) {
    for (uint64_t j = 0; j < J; j++) {
      dc[l * J + j] = l * J + j;
    }
  }
  StridedMemRefType<double, 2> d;
  d.basePtr = d.data = dc.data();
  d.offset = 0;
  d.sizes[0] = L;
  d.sizes[1] = J;
  d.strides[0] = J;
  d.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&d};
    _mlir_ciface_printMemrefF64(&unranked);
  }

  std::vector<double> ac = std::vector<double>(I * J);
  std::fill(ac.begin(), ac.end(), 0.0);

  StridedMemRefType<double, 2> a;
  a.basePtr = a.data = ac.data();
  a.offset = 0;
  a.sizes[0] = I;
  a.sizes[1] = J;
  a.strides[0] = J;
  a.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&a};
    _mlir_ciface_printMemrefF64(&unranked);
  }


  // uint64_t tStartMttkrpCoo = _mlir_ciface_nanoTime();
  sparse_mttkrp(nnz, I, J, K,L, bCoord0, bCoord1, bCoord2, bVals, c, d, a);
  // uint64_t tEndMttkrpCoo = _mlir_ciface_nanoTime();
  // printf("time: %lu\n", tEndMttkrpCoo - tStartMttkrpCoo);

  {
    UnrankedMemRefType<double> unranked{2, (void *)&a};
    _mlir_ciface_printMemrefF64(&unranked);
  }
}