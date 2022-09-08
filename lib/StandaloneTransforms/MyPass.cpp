#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <iomanip>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "StandaloneTransforms/Passes.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "my-pass"

namespace mlir {
    namespace standalone {
#define GEN_PASS_CLASSES

#include "StandaloneTransforms/Passes.h.inc"

    }
}


using namespace mlir;

namespace {
    struct MyPass
            : public mlir::standalone::MyPassBase<MyPass> {
        void runOnOperation() override;
    };
} // end anonymous namespace

namespace mlir {
    namespace standalone {
        std::unique_ptr<OperationPass<>> createMyPass() {
            return std::make_unique<MyPass>();
        }
    }
}

void populateStandaloneToSomethingConversionPatterns(RewritePatternSet &patterns) {
//    patterns.add<>(patterns.getContext());
}

void MyPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateStandaloneToSomethingConversionPatterns(patterns);
    ConversionTarget target(getContext());
    // Here we want to add specific operations, or dialects, that are legal targets for this lowering.
    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
            scf::SCFDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
        signalPassFailure();
}

    LLVM_DEBUG(llvm::dbgs() << "TACO:" << getOperation()->getName() << "\n");
}
