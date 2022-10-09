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
        printf("loop[");
        printf("level:%d,", loop->level_);
        printf("need:%s,", loop->needLoop_ ? "y" : "n");

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
                            printf("over:%s,", var.curr_var()->name().c_str());
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

            mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), zero, m[upper_bound],
                                                                      one);

            // TODO: when we're actually calling the statement we will need to be storing these off somehow.
            forOp.getInductionVar();

            // Start add future loops inside this loop
            builder.setInsertionPointToStart(forOp.getBody());
        } else {
            // This seems to break a relation such as "t8=UF(a,b)" into equality conjuncts: (there will only be one in
            // this case) "t8=UF(a,b)".
            for (omega::EQ_Iterator eq_conj(bounds.single_conjunct()->EQs()); eq_conj; eq_conj++) {
                for (omega::Constr_Vars_Iter var(*eq_conj); var; var++) {
                    // If the current var has an arity, it's a function. No idea what "Global" means in this
                    // circumstance. From something like "t8=UF(a,b)": this code will find "UF(a,b)". From something
                    // like "t8=0" we won't find anything.
                    if (var.curr_var()->kind() == omega::Global_Var && var.curr_var()->get_global_var()->arity() > 0) {
                        printf("uf_call:%s,", var.curr_var()->name().c_str());
                    }
                }
            }
        }

        printf("]\n");
        dispatch(loop->body_); // recurse to next level
    }

    void walkLeaf(omega::CG_leaf *leaf) {
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
    // void mttkrp(int I, int K, int L, int J, double *B,
    //               double *A, double *C, double *D) {
    // for (i = 0; i < I; i++)
    //   for (k = 0; k < K; k++)
    //     for (l = 0; l < L; l++)
    //       for (j = 0; j < J; j++)
    //         A[i,j] += B[i,k,l]*D[l,j]*C[k,j];
    vector<pair<string, string> > dataReads;
    vector<pair<string, string> > dataWrites;
    Computation mttkrp;
    mttkrp.addDataSpace("A", "double*");
    mttkrp.addDataSpace("B", "double*");
    mttkrp.addDataSpace("C", "double*");
    mttkrp.addDataSpace("D", "double*");
    Stmt *s0 = new Stmt("A(i,j) += B(i,k,l)*D(l,j)*C(k,j)",
                        "{[i,k,l,j] : 0<=i<I and 0<=k<K and 0<=l<L and 0<=j<J}",
                        "{[i,k,l,j]->[0,i,0,k,0,l,0,j,0]}",
                        {
                            // data reads
                            {"A", "{[i,k,l,j]->[i,j]}"},
                            {"B", "{[i,k,l,j]->[i,k,l]}"},
                            {"C", "{[i,k,l,j]->[k,j]}"},
                            {"D", "{[i,k,l,j]->[l,j]}"},
                        },
                        {
                            // data writes
                            {"A", "{[i,k,l,j]->[i,j]}"},
                        });

    mttkrp.addStmt(s0);

    // this one is COO
    Computation mttkrp_sps;
    mttkrp_sps.addDataSpace("X", "double*");
    mttkrp_sps.addDataSpace("A", "double*");
    mttkrp_sps.addDataSpace("B", "double*");
    mttkrp_sps.addDataSpace("C", "double*");
    Stmt *s1 = new Stmt("A(x,i,j,k,r) += X(x,i,j,k,r)*B(x,i,j,k,r)*C(x,i,j,k,r)",
                        "{[x,i,j,k,r] :  0<=x< NNZ and i=UFi(x) and "
                        "j=UFj(x) and k=UFk(x) and 0<=r<R}",
                        "{[x,i,j,k,r]->[0,x,0,i,0,j,0,k,0,r,0]}",
                        dataReads,
                        dataWrites);

    mttkrp_sps.addStmt(s1);

    cout << "C sparse codegen ===========================\n";
    cout << mttkrp_sps.codeGen();

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();

    mlir::OpBuilder builder(&context);

    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    cout << "MLIR dense codegen =========================\n";
    // Generate dense code
    builder.setInsertionPointToEnd(theModule.getBody());
    std::unordered_map<std::string, mlir::arith::ConstantIndexOp> m;
    m["I"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 10);
    m["J"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 20);
    m["K"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 30);
    m["R"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 40);
    omega::CG_result *ast = mttkrp.thing();
    Walker(builder, m).walk(ast);

    cout << "MLIR sparse codegen ========================\n";
    // Generate sparse code
    builder.setInsertionPointToEnd(theModule.getBody());
    m["NNZ"] = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 100);
    omega::CG_result *sps_ast = mttkrp_sps.thing();
    Walker(builder, m).walk(sps_ast);

    theModule.dump();

    return 0;
}
