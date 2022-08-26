#include "iegenlib.h"
#include <utility>
#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/InitAllDialects.h>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <omega.h>
#include <code_gen/CG.h>
#include <code_gen/codegen_error.h>
#include <unordered_map>

using iegenlib::Computation;
using namespace std;

class Walker {
public:
    explicit Walker(mlir::OpBuilder builder,
                    std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m) : builder(builder), m(m) {
        // For aesthetic reasons it is nice to only have one zero and 1 constant op created.
        zero = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
        one = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    }

    // This could be done in a nicer way with a visitor pattern, but I don't feel like mucking about in t
    void dispatch(omega::CG_result *t) {
        auto loop = dynamic_cast<omega::CG_loop *>(t);
        auto split = dynamic_cast<omega::CG_split *>(t);
        auto leaf = dynamic_cast<omega::CG_leaf *>(t);
        if (loop) {
            walk(loop);
        } else if (leaf) {
            walk(leaf);
        } else if (split) {
            std::cerr << "err: " << "split not implemented" << std::endl;
            exit(1);
        } else {
            std::cerr << "err: " << "unreachable" << std::endl;
            exit(1);
        }
    }

    void walk(omega::CG_loop *loop) {
        printf("loop ");
        printf("level:%d ", loop->level_);
        printf("need:%s ", loop->needLoop_ ? "y" : "n");
        printf("\n");

        if (loop->needLoop_) {
            std::string up;
            // This seems to break a relation such as "0<=t8 && t8<R" into individual conjuncts: "0<=t8", and "t8<R". The
            // documentation indicates that it will be in a particular form. Not sure if that really matters.
            //
            // GEQ_Iterator overloads ++ operator
            for (omega::GEQ_Iterator conjunct(const_cast<omega::Relation &>(loop->bounds_)
                                                      .single_conjunct()->GEQs()); conjunct; conjunct++) {
                // I don't really know what this is, but I can tell you some things that are true about it: it's `-1` if
                // this conjunct is an upper bound, and `1` if this is a lower bound.
                omega::coef_t coef = (*conjunct).get_coef(const_cast<omega::Relation &>(loop->bounds_)
                                                                  .set_var(loop->level_));
                if (coef == -1) {
                    // This grabs the variable id that this loop will be looping over. If the bounds are "0<=t8 && t8<R"
                    // the variable will be "t8".
                    omega::Variable_ID v = const_cast<omega::Relation &>(loop->bounds_).set_var(loop->level_);

                    // Since the current conjunct should be something like "t8<R" whichever one *isn't* "t8" should be
                    // the loop bound.
                    for (omega::Constr_Vars_Iter cvi(*conjunct); cvi; cvi++) {
                        if (cvi.curr_var() != v) {
                            // TODO: maybe do this with a
                            up = cvi.curr_var()->name();
                            std::cout << cvi.curr_var()->name() << std::endl;
                        }
                    }
                } else if (coef == 1) {
                    // lower bound assumed to be 0 for now
                } else {
                    std::cerr << "err: " << "unreachable" << std::endl;
                    exit(1);
                }
            }

            if (up.empty() || m.find(up) == m.end()) {
                std::cerr << "err: " << "oh no!" << std::endl;
                exit(1);
            }

            // create
            mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), zero, m[up], one);

            // Start add future loops inside this loop
            builder.setInsertionPointToStart(forOp.getBody());
        }

        dispatch(loop->body_);
    }

    void walk(omega::CG_leaf *leaf) {
        printf("leaf\n");
        builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 100);
    }

    void walk(omega::CG_result *t) {
        printf("result\n");
        dispatch(t);
    }

private:
    mlir::OpBuilder builder;
    std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m;
    mlir::arith::ConstantIndexOp zero;
    mlir::arith::ConstantIndexOp one;
};


int main(int argc, char **argv) {
    // this one is dense
    //void mttkrp(int I,int J, int K, int R,double *X,
    //               double *A, double *B, double *C) {
    // for (i = 0; i < I; i++)
    //   for (j = 0; j < J; j++)
    //     for (k = 0; k < K; k++)
    //       for (r = 0; r < R; r++)
    //         A[i,r] += X[i,j,k]*B[j,r]*C[k,r];
    ///
    vector<pair<string, string> > dataReads;
    vector<pair<string, string> > dataWrites;
    Computation mttkrp;
    mttkrp.addDataSpace("X", "double*");
    mttkrp.addDataSpace("A", "double*");
    mttkrp.addDataSpace("B", "double*");
    mttkrp.addDataSpace("C", "double*");
    Stmt *s0 = new Stmt("A(i,r) += X(i,j,k)*B(j,r)*C(k,r)",
                        "{[i,j,k,r] : 0 <= i < I and 0<=j<J and 0<=k<K and 0<=r<R}",
                        "{[i,j,k,r]->[0,i,0,j,0,k,0,r,0]}",
                        dataReads,
                        dataWrites);

    mttkrp.addStmt(s0);

    cout << "Codegen:\n";
    cout << mttkrp.codeGen();

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();

    mlir::OpBuilder builder(&context);

    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(theModule.getBody());

    std::unordered_map<std::string, mlir::arith::ConstantIndexOp> m;
    m["I"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 10);
    m["J"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 20);
    m["K"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 30);
    m["R"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 40);

    Walker(builder, m).walk(mttkrp.thing());

    theModule.dump();

    return 0;
}
