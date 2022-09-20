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

class Thing : public OpRewritePattern<standalone::BarOp> {
public:
    using OpRewritePattern<standalone::BarOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(standalone::BarOp op,
                                  PatternRewriter &rewriter) const override {

        LLVM_DEBUG(llvm::dbgs() << "BarOp >>>>>\n");
        for (auto a : op.getLoc().getContext()->getAvailableDialects()) {
            LLVM_DEBUG(llvm::dbgs() << a << "\n");
        }
        LLVM_DEBUG(llvm::dbgs() << "=========\n");
        for (auto a : op.getLoc().getContext()->getLoadedDialects()) {
            LLVM_DEBUG(llvm::dbgs() << a->getNamespace() << "\n");
        }
        LLVM_DEBUG(llvm::dbgs() << "<<<<< BarOp\n");
        mlir::arith::ConstantIndexOp  zero = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        mlir::arith::ConstantIndexOp one = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 1);
        mlir::arith::ConstantIndexOp ten = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 10);


        auto forOp = rewriter.replaceOpWithNewOp<scf::ForOp>(op, zero, ten, one);
        auto indVar = forOp.getInductionVar();


        return success();
    }
};

void populateStandaloneToSomethingConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<Thing>(patterns.getContext());
}

void MyPass::runOnOperation() {
    auto ctx = &getContext();
    for (auto dialect : ctx->getDialectRegistry().getDialectNames()) {
        LLVM_DEBUG(llvm::dbgs() << dialect << "\n");
    }
    RewritePatternSet patterns(&getContext());
    populateStandaloneToSomethingConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<scf::SCFDialect, arith::ArithmeticDialect>();
    // Here we want to add specific operations, or dialects, that are legal targets for this lowering.
    target.addIllegalOp<standalone::BarOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "TACO:" << getOperation()->getName() << "\n");
}
