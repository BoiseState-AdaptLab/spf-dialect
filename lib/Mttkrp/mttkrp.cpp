#include "iegenlib.h"
#include <utility>
#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/InitAllDialects.h>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

using iegenlib::Computation;
using namespace std;

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
    mlir::OpBuilder builder(&context);

    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(theModule.getBody());

    llvm::SmallVector<mlir::Value, 2> lbs = {
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0),
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0),
    };
    llvm::SmallVector<mlir::Value, 2> ubs = {
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 10),
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 10),
    };
    llvm::SmallVector<mlir::Value, 2> steps = {
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1),
            builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1),
    };
    mlir::scf::buildLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps);

    theModule.dump();

    return 0;
}
