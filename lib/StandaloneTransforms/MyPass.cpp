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
#include <vector>
#include <unordered_map>
#include "iegenlib.h"
#include <omega.h>
#include <code_gen/CG.h>
#include <code_gen/codegen_error.h>
#include <llvm/ADT/Optional.h>

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


namespace {
    class Walker {
    public:
        explicit Walker(mlir::OpBuilder &builder, standalone::BarOp &barOp,
                        std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m) : builder(builder), m(m),
                                                                                            barOp(barOp) {
            // For aesthetic reasons it is nice to only have one zero and 1 constant op created.
            zero = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 0);
            one = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 1);
        }

        llvm::Optional<mlir::scf::ForOp> walk(omega::CG_result *t) {
            LLVM_DEBUG(llvm::dbgs() << "result\n");
            dispatch(t);
            return maybeOp;
        }

    private:
        mlir::OpBuilder &builder;
        std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m;
        mlir::arith::ConstantIndexOp zero;
        mlir::arith::ConstantIndexOp one;
        llvm::Optional<mlir::scf::ForOp> maybeOp;
        standalone::BarOp &barOp;
        std::vector<mlir::Value> ivs; // induction variables

        // This could be done in a nicer way with a visitor pattern, but I don't feel like mucking about in t
        void dispatch(omega::CG_result *t) {
            auto loop = dynamic_cast<omega::CG_loop *>(t);
            auto split = dynamic_cast<omega::CG_split *>(t);
            auto leaf = dynamic_cast<omega::CG_leaf *>(t);
            if (loop) {
                walkLoop(loop);
            } else if (leaf) {
                walkLeaf(leaf);
            } else if (split) {
                std::cerr << "err: " << "split not implemented" << std::endl;
                exit(1);
            } else {
                std::cerr << "err: " << "unreachable" << std::endl;
                exit(1);
            }
        }

        void walkLoop(omega::CG_loop *loop) {
            LLVM_DEBUG(llvm::dbgs() << "loop[");
            LLVM_DEBUG(llvm::dbgs() << "level:" << loop->level_ << ",");
            LLVM_DEBUG(llvm::dbgs() << "need:" << (loop->needLoop_ ? "y" : "n") << ",");

            auto bounds = const_cast<omega::Relation &>(loop->bounds_);

            // Loops will be created for each level in the execution schedule. Some levels will require a loop to be
            // generated, some a call to an uninterpreted function, some don't require any code to be generated.
            if (loop->needLoop_) {
                // (Should be) set while looping over greater than or equal to conjuncts.
                std::string upper_bound;

                // This seems to break a relation such as "0<=t8 && t8<R" into individual greater than or equal to
                // conjuncts: "0<=t8", and "t8<R".
                for (omega::GEQ_Iterator geq_conj(bounds.single_conjunct()->GEQs()); geq_conj; geq_conj++) {
                    // bounds.set_var grabs the induction variable for the current loop. If the bounds are "0<=t8 && t8<R"
                    // the variable will be "t8".
                    omega::Variable_ID induction_variable = bounds.set_var(loop->level_);
                    // I don't really know what this is, but I can tell you some things that are true about it: it's `-1` if
                    // this geq_conj is an upper bound, and `1` if this is a lower bound.
                    omega::coef_t coef = (*geq_conj).get_coef(induction_variable);
                    if (coef == -1) {
                        // The current geq_conj should be something like "t8<R". Whichever variable in the conjunct *isn't*
                        // "t8" should be the loop bound.
                        for (omega::Constr_Vars_Iter var(*geq_conj); var; var++) {
                            if (var.curr_var() != induction_variable) {
                                upper_bound = var.curr_var()->name();
                                LLVM_DEBUG(llvm::dbgs() << "over:" << var.curr_var()->name() << ",");
                            }
                        }
                    } else if (coef == 1) {
                        // lower bound assumed to be 0 for now
                    } else {
                        std::cerr << "err: " << "unreachable" << std::endl;
                        exit(1);
                    }
                }

                if (upper_bound.empty() || m.find(upper_bound) == m.end()) {
                    std::cerr << "err: " << "oh no!" << std::endl;
                    exit(1);
                }

                mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(barOp.getLoc(), zero, m[upper_bound],
                                                                          one);
                // if this is the top level loop store it off to return
                if (!maybeOp) {
                    maybeOp = forOp;
                }

                // store off induction variable
                ivs.push_back(forOp.getInductionVar());

                // Start add future loops inside this loop
                builder.setInsertionPointToStart(forOp.getBody());
            } else { // non loops may require a call to a UF
                // This seems to break a relation such as "t8=UF(a,b)" into equality conjuncts: (there will only be one in
                // this case) "t8=UF(a,b)".
                for (omega::EQ_Iterator eq_conj(bounds.single_conjunct()->EQs()); eq_conj; eq_conj++) {
                    for (omega::Constr_Vars_Iter var(*eq_conj); var; var++) {
                        // If the current var has an arity, it's a function. No idea what "Global" means in this
                        // circumstance. From something like "t8=UF(a,b)": this code will find "UF(a,b)". From something
                        // like "t8=0" we won't find anything.
                        if (var.curr_var()->kind() == omega::Global_Var &&
                            var.curr_var()->get_global_var()->arity() > 0) {
                            LLVM_DEBUG(llvm::dbgs() << "uf_call:" << var.curr_var()->name() << ",");
                        }
                    }
                }
            }

            LLVM_DEBUG(llvm::dbgs() << "]\n");
            dispatch(loop->body_); // recurse to next level
        }

        void walkLeaf(omega::CG_leaf *leaf) {
            LLVM_DEBUG(llvm::dbgs() << "leaf\n");

            // TODO: there's no validation that the region has a block.
            auto &block = barOp->getRegion(0).front();

            // BlockAndValueMapping holds a mapping between values.
            BlockAndValueMapping map;
            // TODO: I'm not sure it's true (actually I'm almost positive it isn't) that the original induction variable
            // order will be the same after code generation.
            map.map(block.getArguments(), ivs);
            for (auto &op: block) {
                // clone creates the op with the map applied.
                builder.clone(op, map);
            }
        }
    };


    class ReplaceWithCodeGen : public OpRewritePattern<standalone::BarOp> {
    public:
        using OpRewritePattern<standalone::BarOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(standalone::BarOp barOp, PatternRewriter &rewriter) const override {
            // Dense mttkrp
            //
            // void mttkrp(int I,int J, int K, int R,double *X,
            //               double *A, double *B, double *C) {
            //   for (i = 0; i < I; i++)
            //     for (j = 0; j < J; j++)
            //       for (k = 0; k < K; k++)
            //         for (r = 0; r < R; r++)
            //           A[i,r] += X[i,j,k]*B[j,r]*C[k,r];
            std::vector<std::pair<std::string, std::string> > dataReads;
            std::vector<std::pair<std::string, std::string> > dataWrites;
            Computation mttkrp;
            mttkrp.addDataSpace("X", "double*");
            mttkrp.addDataSpace("A", "double*");
            mttkrp.addDataSpace("B", "double*");
            mttkrp.addDataSpace("C", "double*");
            Stmt *s0 = new Stmt("A(i,r) += X(i,j,k)*B(j,r)*C(k,r)",
                                "{[i,j,k,r] : 0 <= i < I and 0<=j<J and 0<=k<K and 0<=r<R}",
                                "{[i,j,k,r]->[0,i,0,j,0,k,0,r,0]}",
                                {
                                        // data reads
                                        {"A", "{[i,k,l,j]->[i,j]}"},
                                        {"B", "{[i,k,l,j]->[i,k,l]}"},
                                        {"D", "{[i,k,l,j]->[l,j]}"},
                                        {"C", "{[i,k,l,j]->[k,j]}"},
                                },
                                {
                                        // data writes
                                        {"A", "{[i,k,l,j]->[i,j]}"},
                                });

            mttkrp.addStmt(s0);
            LLVM_DEBUG({
                           llvm::dbgs() << "C dense codegen ===========================\n";
                           llvm::dbgs() << mttkrp.codeGen();
                           llvm::dbgs() << "===========================================\n";
                       });


            std::unordered_map<std::string, mlir::arith::ConstantIndexOp> m;
            m["I"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 10);
            m["J"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 20);
            m["K"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 30);
            m["R"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 40);
            omega::CG_result *ast = mttkrp.thing();
            auto loop = Walker(rewriter, barOp, m).walk(ast);
            if (!loop) {
                return failure();
            }
            rewriter.eraseOp(barOp);
            return success();
        }
    };
} // end anonymous namespace

void populateStandaloneToSomethingConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<ReplaceWithCodeGen>(patterns.getContext());
}

void MyPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateStandaloneToSomethingConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<scf::SCFDialect, arith::ArithmeticDialect, vector::VectorDialect>();
    target.addIllegalOp<standalone::BarOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
        signalPassFailure();
    }
}
