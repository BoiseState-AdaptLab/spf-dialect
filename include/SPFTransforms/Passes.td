#ifndef MLIR_TRANSFORMS_PASSES
#define MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConvertSPFToLoops : Pass<"convert-spf-to-loops"> {
    let summary = "Lowers spf dialect to scf loops.";
    let description = [{
        See section 3.3 "lowering" of thesis: "Portable Sparse Polyhedral
        Framework code generation using Multi Level Intermediate Representation"
        by Aaron St George at Boise State University for description of lowering.
    }];
    let constructor = "mlir::spf::createConvertSPFToLoops()";
    let dependentDialects = ["mlir::gpu::GPUDialect"];
}

#endif // MLIR_TRANSFORMS_PASSES
