#include "Runtime/runtime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

void iegenlibMTTKRP(int NNZ, int J, uint64_t *UFi, uint64_t *UFk, uint64_t *UFl,
                    double *values, double *c, double *d, double *a) {
#define A(i, j) a[i * J + j]
#define B(z) values[z]
#define D(l, j) d[l * J + j]
#define C(k, j) c[k * J + j]

  uint64_t t1, t2, t3, t4, t5;

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
}

enum test {
  test_mlir,
  test_iegen,
};

// Sparse MTTKRP: http://tensor-compiler.org/docs/data_analytics
void mttkrp(test which, char *filename) {
  COO *coo = (COO *)_mlir_ciface_read_coo(filename);
  assert(coo->rank == 3 && "mttkrp requires rank 3 tensor");

  // These functions look weird because they are also used as a runtime
  // functions called by MLIR code. I don't fully understand it but they have to
  // look a certain way.
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
  uint64_t NNZ = coo->nnz;

  // Construct c matrix
  std::vector<double> cData = std::vector<double>(K * J);
  for (uint64_t k = 0; k < K; k++) {
    for (uint64_t j = 0; j < J; j++) {
      cData[k * J + j] = k * J + j;
    }
  }
  StridedMemRefType<double, 2> c;
  c.basePtr = c.data = cData.data();
  c.offset = 0;
  c.sizes[0] = K;
  c.sizes[1] = J;
  c.strides[0] = J;
  c.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&c};
    _mlir_ciface_printMemrefF64(&unranked);
  }

  // Construct d matrix
  std::vector<double> dData = std::vector<double>(L * J);
  for (uint64_t l = 0; l < L; l++) {
    for (uint64_t j = 0; j < J; j++) {
      dData[l * J + j] = l * J + j;
    }
  }
  StridedMemRefType<double, 2> d;
  d.basePtr = d.data = dData.data();
  d.offset = 0;
  d.sizes[0] = L;
  d.sizes[1] = J;
  d.strides[0] = J;
  d.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&d};
    _mlir_ciface_printMemrefF64(&unranked);
  }

  std::vector<double> aData = std::vector<double>(I * J);
  std::fill(aData.begin(), aData.end(), 0.0);

  StridedMemRefType<double, 2> a;
  a.basePtr = a.data = aData.data();
  a.offset = 0;
  a.sizes[0] = I;
  a.sizes[1] = J;
  a.strides[0] = J;
  a.strides[1] = 1;

  {
    UnrankedMemRefType<double> unranked{2, (void *)&a};
    _mlir_ciface_printMemrefF64(&unranked);
  }

  switch (which) {
  case test_mlir:
    sparse_mttkrp(NNZ, I, J, K, L, bCoord0, bCoord1, bCoord2, bVals, c, d, a);
    break;
  case test_iegen:
    iegenlibMTTKRP(NNZ, J, bCoord0.data, bCoord1.data, bCoord2.data, bVals.data,
                   c.data, d.data, a.data);
    break;
  }

  // uint64_t tStartMttkrpCoo = _mlir_ciface_nanoTime();
  // uint64_t tEndMttkrpCoo = _mlir_ciface_nanoTime();
  // printf("time: %lu\n", tEndMttkrpCoo - tStartMttkrpCoo);

  {
    UnrankedMemRefType<double> unranked{2, (void *)&a};
    _mlir_ciface_printMemrefF64(&unranked);
  }
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

  // read b tensor from file
  char *filename = getTensorFilename(0);
  printf("MLIR OUTPUT =============================\n");
  mttkrp(test_mlir, filename);
  printf("IEGEN OUTPUT ============================\n");
  mttkrp(test_iegen, filename);
}