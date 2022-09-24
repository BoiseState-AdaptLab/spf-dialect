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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

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

    LogicalResult matchAndRewrite(standalone::BarOp barOp, PatternRewriter &rewriter) const override {
        mlir::arith::ConstantIndexOp zero = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 0);
        mlir::arith::ConstantIndexOp one = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 1);
        mlir::arith::ConstantIndexOp ten = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 10);

        auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
            auto &block = barOp->getRegion(0).front();
            BlockAndValueMapping map;
            map.map(block.getArguments(), ivs);
            for (auto &op: block) {
                rewriter.clone(op, map);
            }

            b.create<mlir::scf::YieldOp>(barOp.getLoc());
        };
        rewriter.replaceOpWithNewOp<scf::ForOp>(barOp, zero, ten, one, llvm::None, bodyBuilder);

        return success();
    }
};

void populateStandaloneToSomethingConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<Thing>(patterns.getContext());
}

void MyPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateStandaloneToSomethingConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<scf::SCFDialect, arith::ArithmeticDialect, vector::VectorDialect>();
    // Here we want to add specific operations, or dialects, that are legal targets for this lowering.
    target.addIllegalOp<standalone::BarOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "TACO:" << getOperation()->getName() << "\n");
}
