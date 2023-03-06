#ifndef SPF_SPFOPS_H
#define SPF_SPFOPS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SPF/SPFOps.h.inc"

#endif // SPF_SPFOPS_H
