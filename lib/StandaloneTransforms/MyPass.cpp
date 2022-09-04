#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <iomanip>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "StandaloneTransforms/Passes.h"

#define DEBUG_TYPE "my-pass"

namespace mlir {
    namespace standalone {
#define GEN_PASS_CLASSES

#include "StandaloneTransforms/Passes.h.inc"

    }
}

namespace {
/// Loop invariant code motion (LICM) pass.
    struct MyPass
            : public mlir::standalone::MyPassBase<MyPass> {
        void runOnOperation() override;
    };
} // end anonymous namespace

namespace mlir {
    namespace standalone {
        std::unique_ptr<OperationPass<func::FuncOp>> createMyPass() {
            return std::make_unique<MyPass>();
        }
    }
}

void MyPass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << "TACO: Hello from my pass!\n");
}
