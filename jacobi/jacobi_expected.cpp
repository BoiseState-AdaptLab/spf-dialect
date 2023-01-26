#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

int main(int argc, char *argv[]) {
  int ub_T = 10;
  int ub_x = 8;
  int lb_x = 1;
  int i;

  std::vector<double> AData(10);
  std::vector<double> BData(10);
  // set up initial conditions
  for (int i = 0; i < 10; ++i) {
    AData[i] = 0.0;
    BData[i] = 0.0;
  }
  AData[9] = 100.0;
  BData[9] = 100.0;

  StridedMemRefType<double, 1> A;
  A.basePtr = A.data = AData.data();
  A.offset = 0;
  A.sizes[0] = 10;
  A.strides[0] = 1;
  StridedMemRefType<double, 1> B;
  B.basePtr = B.data = BData.data();
  B.offset = 0;
  B.sizes[0] = 10;
  B.strides[0] = 1;

  // run jacobian
  for (int t = 1; t <= ub_T / 2; ++t) {
    for (i = lb_x; i <= ub_x; ++i)
      A[i] = (B[i - 1] + B[i] + B[i + 1]) / 3.0;

    for (i = lb_x; i <= ub_x; ++i)
      B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0;
  }

  impl::printMemRef(A);
  impl::printMemRef(B);
}
