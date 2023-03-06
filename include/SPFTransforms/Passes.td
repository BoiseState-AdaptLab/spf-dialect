#ifndef MLIR_TRANSFORMS_PASSES
#define MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MyPass : Pass<"my-pass"> {
  let summary = "prints some debug stuff";
  let description = [{
      This pass prints some debug stuff
  }];
  let constructor = "mlir::spf::createMyPass()";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

#endif // MLIR_TRANSFORMS_PASSES
