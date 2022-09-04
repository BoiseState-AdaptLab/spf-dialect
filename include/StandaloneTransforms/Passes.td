#ifndef MLIR_TRANSFORMS_PASSES
#define MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MyPass : Pass<"my-pass", "mlir::func::FuncOp"> {
  let summary = "prints some debug stuff";
  let description = [{
      This pass prints some debug stuff
  }];
  let constructor = "mlir::standalone::createMyPass()";
}

#endif // MLIR_TRANSFORMS_PASSES
