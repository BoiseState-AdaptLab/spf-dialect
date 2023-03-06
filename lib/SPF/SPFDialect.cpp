#include "SPF/SPFDialect.h"
#include "SPF/SPFOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::spf;

#include "SPF/SPFOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SPF dialect.
//===----------------------------------------------------------------------===//

void SPFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SPF/SPFOps.cpp.inc"
      >();
}
