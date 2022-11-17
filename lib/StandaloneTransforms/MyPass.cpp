#include "Printer.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "StandaloneTransforms/Passes.h"
#include "iegenlib.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "set_relation/set_relation.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <code_gen/CG.h>
#include <code_gen/codegen_error.h>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <omega.h>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define DEBUG_TYPE "my-pass"

using ReadWrite = std::vector<std::pair<std::string, std::string>>;

namespace mlir {
namespace standalone {
#define GEN_PASS_CLASSES

#include "StandaloneTransforms/Passes.h.inc"

} // namespace standalone
} // namespace mlir

using namespace mlir;

namespace {
struct MyPass : public mlir::standalone::MyPassBase<MyPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

namespace mlir {
namespace standalone {
std::unique_ptr<OperationPass<>> createMyPass() {
  return std::make_unique<MyPass>();
}
} // namespace standalone
} // namespace mlir

namespace {

// Copy pasted from Loops.cpp (called as part of --convert-linalg-to-loops)
//
// makeCanonicalAffineApplies produces `affine.apply` ops that apply indexing
// maps to induction variables to produce index variables. For example: this
// code generates `affine.apply`s (1), (2), and (3) to apply the map:
// affine_map<(i,k,l,j)->(i,k,l)>. The generated index variables are later used
// to load from a memref<?x?x?xf64> at [%0,%1,%2].
//
// #map0 = affine_map<(d0)-> (d0)>
// scf.for %i = %c0 to %c2 step %c1 {
//   scf.for %k = %c0 to %c3 step %c1 {
//     scf.for %l = %c0 to %c4 step %c1 {
//       scf.for %j = %c0 to %c5 step %c1 {
//         %0 = affine.apply #map0(%i)        (1)
//         %1 = affine.apply #map0(%k)        (2)
//         %2 = affine.apply #map0(%l)        (3)
static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &builder,
                                                     Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());

  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(builder.create<AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

struct Walker {
public:
  explicit Walker(
      mlir::OpBuilder &builder, standalone::BarOp &barOp,
      std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m,
      llvm::Optional<mlir::AffineMap> inverseMap =
          llvm::Optional<mlir::AffineMap>())
      : builder(builder), m(m), barOp(barOp), inverseMap(inverseMap) {
    // For aesthetic reasons it is nice to only have one zero and 1 constant op
    // created.
    zero = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 0);
    one = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 1);
  }

  llvm::Optional<mlir::scf::ForOp> walk(omega::CG_result *t) {
    LLVM_DEBUG(llvm::dbgs() << "Walker ====================================\n");
    LLVM_DEBUG(llvm::dbgs() << "result\n");
    dispatch(t);
    return maybeOp;
  }

  mlir::OpBuilder &builder;
  std::unordered_map<std::string, mlir::arith::ConstantIndexOp> &m;
  mlir::arith::ConstantIndexOp zero;
  mlir::arith::ConstantIndexOp one;
  llvm::Optional<mlir::scf::ForOp> maybeOp;
  standalone::BarOp &barOp;
  llvm::Optional<mlir::AffineMap> inverseMap;
  std::vector<mlir::Value> ivs; // induction variables

  // This could be done in a nicer way with a visitor pattern, but I don't feel
  // like mucking about in t
  void dispatch(omega::CG_result *t) {
    auto loop = dynamic_cast<omega::CG_loop *>(t);
    auto split = dynamic_cast<omega::CG_split *>(t);
    auto leaf = dynamic_cast<omega::CG_leaf *>(t);
    if (loop) {
      walkLoop(loop);
    } else if (leaf) {
      walkLeaf(leaf);
    } else if (split) {
      std::cerr << "err: "
                << "split not implemented" << std::endl;
      exit(1);
    } else {
      std::cerr << "err: "
                << "unreachable" << std::endl;
      exit(1);
    }
  }

  void walkLoop(omega::CG_loop *loop) {
    LLVM_DEBUG(llvm::dbgs() << "loop[");
    LLVM_DEBUG(llvm::dbgs() << "level:" << loop->level_ << ",");
    LLVM_DEBUG(llvm::dbgs() << "need:" << (loop->needLoop_ ? "y" : "n") << ",");

    auto bounds = const_cast<omega::Relation &>(loop->bounds_);

    // Loops will be created for each level in the execution schedule. Some
    // levels will require a loop to be generated, some a call to an
    // uninterpreted function, some don't require any code to be generated.
    if (loop->needLoop_) {
      // (Should be) set while looping over greater than or equal to conjuncts.
      std::string upper_bound;

      // This seems to break a relation such as "0<=t8 && t8<R" into individual
      // greater than or equal to conjuncts: "0<=t8", and "t8<R".
      for (omega::GEQ_Iterator geq_conj(bounds.single_conjunct()->GEQs());
           geq_conj; geq_conj++) {
        // bounds.set_var grabs the induction variable for the current loop. If
        // the bounds are "0<=t8 && t8<R" the variable will be "t8".
        omega::Variable_ID induction_variable = bounds.set_var(loop->level_);
        // I don't really know what this is, but I can tell you some things that
        // are true about it: it's `-1` if this geq_conj is an upper bound, and
        // `1` if this is a lower bound.
        omega::coef_t coef = (*geq_conj).get_coef(induction_variable);
        if (coef == -1) {
          // The current geq_conj should be something like "t8<R". Whichever
          // variable in the conjunct *isn't* "t8" should be the loop bound.
          for (omega::Constr_Vars_Iter var(*geq_conj); var; var++) {
            if (var.curr_var() != induction_variable) {
              upper_bound = var.curr_var()->name();
              LLVM_DEBUG(llvm::dbgs()
                         << "over:" << var.curr_var()->name() << ",");
            }
          }
        } else if (coef == 1) {
          // lower bound assumed to be 0 for now
        } else {
          std::cerr << "err: "
                    << "unreachable" << std::endl;
          exit(1);
        }
      }

      if (upper_bound.empty() || m.find(upper_bound) == m.end()) {
        std::cerr << "err: "
                  << "oh no!" << std::endl;
        exit(1);
      }

      mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(
          barOp.getLoc(), zero, m[upper_bound], one);
      // if this is the top level loop store it off to return
      if (!maybeOp) {
        maybeOp = forOp;
      }

      // store off induction variable
      ivs.push_back(forOp.getInductionVar());

      // Start add future loops inside this loop
      builder.setInsertionPointToStart(forOp.getBody());
    } else { // non loops may require a call to a UF
      // This seems to break a relation such as "t8=UF(a,b)" into equality
      // conjuncts: (there will only be one in this case) "t8=UF(a,b)".
      for (omega::EQ_Iterator eq_conj(bounds.single_conjunct()->EQs()); eq_conj;
           eq_conj++) {
        for (omega::Constr_Vars_Iter var(*eq_conj); var; var++) {
          // If the current var has an arity, it's a function. No idea what
          // "Global" means in this circumstance. From something like
          // "t8=UF(a,b)": this code will find "UF(a,b)". From something like
          // "t8=0" we won't find anything.
          if (var.curr_var()->kind() == omega::Global_Var &&
              var.curr_var()->get_global_var()->arity() > 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "uf_call:" << var.curr_var()->name() << ",");
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "]\n");
    dispatch(loop->body_); // recurse to next level
  }

  // fixMap composes a map from iteration space tuple to something with the
  // inverse function for any transformations done to the iteration space to get
  // to the execution schedule. Loops are generated from the execution schedule.
  AffineMap fixMap(AffineMap fromIterationSpace) {
    if (inverseMap) {
      fromIterationSpace = fromIterationSpace.compose(*inverseMap);
    }
    return fromIterationSpace;
  }

  /// Much of this function adapted from `emitScalarImplementation` fuction in
  /// Loops.cpp in mlir at sha: 657bfa. Loops.cpp implements the bulk of the
  /// --convert-linalg-to-loops pass.
  ///
  /// This function does three things:
  ///   1. Emitting load ops from each input by applying the appropriate
  ///      input or output map to the induction variables generated by
  ///      polyhedral scanning.
  ///   2. Inlining the statement that takes as arguments the scalars
  ///      from point 1. above.
  ///   3. Emitting store ops to store the results of 2. to the output
  ///      views.
  ///
  /// An example output may resemble:
  ///
  /// ```
  ///    scf.for %i = %c0 to %0 step %c1 {
  ///      scf.for %j = %c0 to %1 step %c1 {
  ///        scf.for %k = %c0 to %4 step %c1 {
  ///          %11 = load %arg0[%i, %j] :
  ///            memref<?x?xf32, stride_specification>
  ///          %12 = load %arg1[%i, %j, %k] :
  ///            memref<?x?x?xf32, stride_specification>
  ///          %13 = load %arg2[%i, %k, %j] :
  ///            memref<?x?x?xf32, stride_specification>
  ///
  ///          %14 = call @foo(%11, %12, %13) : (f32, f32, f32) -> (f32)
  ///
  ///          store %14, %arg1[%i, %j, %k] :
  ///            memref<?x?x?Xf32, stride_specification>
  ///        }
  ///      }
  ///    }
  /// ```
  void walkLeaf(omega::CG_leaf *leaf) {
    LLVM_DEBUG(llvm::dbgs() << "leaf\n");

    auto loc = barOp->getLoc();

    // TODO: apply fixMap with llvm::map_range(
    auto readMaps = barOp.getReads().getAsValueRange<AffineMapAttr>();
    auto writeMaps = barOp.getWrites().getAsValueRange<AffineMapAttr>();

    SmallVector<Value> indexedValues;
    indexedValues.reserve(barOp->getNumOperands());
    { // 1.a. produce loads from input memrefs
      SmallVector<Value> inputOperands = barOp.getInputOperands();
      for (size_t i = 0; i < inputOperands.size(); i++) {
        // read the map that corresponds with the current inputOperand. It seems
        // like this would be better using the subscript "[]" operator, but
        // indexingMaps doesn't provide one.
        auto map = fixMap(*(readMaps.begin() + i));
        auto indexing = makeCanonicalAffineApplies(builder, loc, map, ivs);
        indexedValues.push_back(
            builder.create<memref::LoadOp>(loc, inputOperands[i], indexing));
      }
    }

    // 1.b. Emit load for output memrefs
    SmallVector<Value> outputOperands = barOp.getOutputOperands();
    for (size_t i = 0; i < outputOperands.size(); i++) {
      // read the map that corresponds with the current inputOperand.
      AffineMap map = fixMap(*(writeMaps.begin() + i));

      SmallVector<Value> indexing =
          makeCanonicalAffineApplies(builder, loc, map, ivs);
      indexedValues.push_back(
          builder.create<memref::LoadOp>(loc, outputOperands[i], indexing));
    }

    // TODO: there's no validation that the region has a block.
    auto &block = barOp.getBody().front();

    // 2. inline statement
    BlockAndValueMapping map; // holds a mapping between values.
    map.map(block.getArguments(), indexedValues);
    for (auto &op : block.without_terminator()) {
      // clone creates the op with the map applied.
      builder.clone(op, map);
    }

    // 3. emit store
    SmallVector<SmallVector<Value>, 8> indexing;
    SmallVector<Value> outputBuffers;
    for (size_t i = 0; i < outputOperands.size(); i++) {
      // read the map that corresponds with the current inputOperand.
      AffineMap map = fixMap(*(writeMaps.begin() + i));
      indexing.push_back(makeCanonicalAffineApplies(builder, loc, map, ivs));
      outputBuffers.push_back(outputOperands[i]);
    }
    Operation *terminator = block.getTerminator();
    for (OpOperand &operand : terminator->getOpOperands()) {
      Value toStore = map.lookupOrDefault(operand.get());
      builder.create<memref::StoreOp>(loc, toStore,
                                      outputBuffers[operand.getOperandNumber()],
                                      indexing[operand.getOperandNumber()]);
    }
  }
};

/// relationForOperand builds an IEGenLib Relation string representation from an
/// AffineMap
///
/// I think it's a little weird to create an IEGenLib relation string from
/// AffineMap. We could probably just build a Relations from an AffineMaps
/// pretty easily without going through a string intermediary. But, it appears
/// to me that the string constructor on Relation is "the API". So OK.
std::string relationForOperand(AffineMap map) {
  std::string read;
  llvm::raw_string_ostream ss(read);
  CopyPastedPrintingStuff(ss).printAffineMap(map);
  return read;
};

class ReplaceWithCodeGen : public OpRewritePattern<standalone::BarOp> {
public:
  using OpRewritePattern<standalone::BarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::BarOp barOp,
                                PatternRewriter &rewriter) const override {
    // Build up reads and writes
    ReadWrite reads;
    ReadWrite writes;
    { // create reads for inputs
      auto readMaps = barOp.getReads().getAsValueRange<AffineMapAttr>();
      for (size_t i = 0; i < barOp.getInputOperands().size(); i++) {
        AffineMap map = *(readMaps.begin() + i);
        std::string read = relationForOperand(map);

        // We won't actually use the data space names for anything, just make
        // something nice-ish for debugging purposes.
        char name[100];
        std::sprintf(name, "input_%zu", i);
        reads.push_back({name, read});
      }
    }
    { // create reads/writes for outputs
      auto writeMaps = barOp.getWrites().getAsValueRange<AffineMapAttr>();
      for (size_t i = 0; i < barOp.getOutputOperands().size(); i++) {
        AffineMap map = *(writeMaps.begin() + i);
        // The += operator counts as both read and write.
        std::string read_write = relationForOperand(map);

        char name[100];
        std::sprintf(name, "output_%zu", i);
        reads.push_back({name, read_write});
        writes.push_back({name, read_write});
      }
    }

    // http://tensor-compiler.org/docs/data_analytics
    // void mttkrp(int I, int K, int L, int J, double *B,
    //               double *A, double *C, double *D) {
    // for(int i = 0; i < I; i++)
    //   for(int k = 0; k < K; k++)
    //     for(int l = 0; l < L; l++)
    //       for(int j = 0; j < J; j++)
    //         A[i,j] += B[i,k,l]*D[l,j]*C[k,j];
    Computation mttkrp;
    mttkrp.addDataSpace("B", "double*");
    mttkrp.addDataSpace("A", "double*");
    mttkrp.addDataSpace("C", "double*");
    mttkrp.addDataSpace("D", "double*");
    Stmt *s0 = new Stmt("A(i,j) += B(i,k,l)*D(l,j)*C(k,j)",
                        "{[i,k,l,j] : 0<=i<I and 0<=k<K and 0<=l<L and 0<=j<J}",
                        "{[i,k,l,j]->[i,k,l,j]}", reads, writes);
    mttkrp.addStmt(s0);
    LLVM_DEBUG({
      llvm::dbgs() << "C dense codegen ===========================\n";
      llvm::dbgs() << mttkrp.codeGen();
    });

    LLVM_DEBUG(llvm::dbgs() << "Adding fake transformation ================\n");

    LLVM_DEBUG(llvm::dbgs() << "transform: {[i,k,l,j] -> [k,i,l,j]}"
                            << "\n");
    auto transform = new Relation("{[i,k,l,j] -> [k,i,l,j]}");
    mttkrp.addTransformation(0, transform);

    // TODO: AffineMap has an inverse fuction. Transformations might be applied
    // as AffineMaps. This could all just be replaced with that.
    //
    // Here we're creating an inverse relation to map transformed execution
    // schedule tuple back to original iteration space tuple.
    //
    // For an example of why we need to do this, assume we have a double for
    // loop:
    //   for(int i=0; i<I; i++)
    //     for(int j=0; j<J; j++)
    //       S0
    // In the polyhedral model this would be:
    //   Iteration space: {[i,j]: 0<=i<I && 0<=j<J}
    //   Execution schedule: {[i,j] -> [i,j]}
    // Take I = 1 and J=2, generated code will execute the iteration space
    // tuples in the order determined by the execution schedule tuples:
    //   | Execution Schedule tuple | Iteration space tuple
    //   | [0,0]                    | [0,0]
    //   | [0,1]                    | [0,1]
    //   | [0,2]                    | [0,2]
    //   | [1,0]                    | [1,0]
    //   | [1,1]                    | [1,1]
    //   | [1,2]                    | [1,2]
    // If we apply a permute transformation: {[i,j] -> [j,i]} the order will
    // change:
    //   | Execution Schedule tuple | Iteration space tuple
    //   | [0,0]                    | [0,0]
    //   | [0,1]                    | [1,0]
    //   | [1,0]                    | [0,1]
    //   | [1,1]                    | [1,1]
    //   | [2,0]                    | [0,2]
    //   | [2,1]                    | [1,2]
    // Notice that we have to map from the new execution schedule tuple back to
    // the original iteration space tuple. For example, [2,1] might index
    // something out of bounds if 2 was used for i and 1 for j. When we're
    // writing regular code, variable binding does this job for us:
    //   for(int i=0; i<I; i++)
    //     for(int j=0; j<J; j++)
    //       A[i,j] = B[i,j]
    // Vs.
    //   for(int j=0; j<J; j++)
    //     for(int i=0; i<I; i++)
    //       A[i,j] = B[i,j]
    // will execute the same statements just in a different order. But the
    // polyhedral model's representation is more abstract. Statements only know
    // about iteration space variables and don't understand how those variables
    // will be mapped to actual induction variables. Since codegen creates
    // induction variables that range over the execution schedule, we need to
    // create a inverse function that undoes whatever transformations have taken
    // place between the original iteration space and the execution schedule to
    // ensure we execute statements with the right induction variables.
    //
    // To construct the inverse relation we must construct a LHS and RHS.
    // For the LHS:
    //   since the inverse relation executes on the output of the relation we're
    //   inverting we simply apply the relation we're inverting to its input.
    //   In our example, we would generate the LHS for our permute relation
    //   inverse by applying the original relation to [i,j] and producing:
    //     [j,i]
    // For th RHS:
    //   It's a little trickier, under the hood the relation we're inverting:
    //     {[i,j]->[j,i]}
    //   is really something like:
    //     {[t0,t1]->[t2,t3]: t0=t3 && t1=t2}.
    //   To construct the RHS, we go through each of the variables in the the
    //   order of input to the relation we're inverting (t0, t1) and ask for a
    //   function solving for each variable using using only the variables in
    //   the output arity (t2, t3). Since the output of the relation we're
    //   inverting is the LHS of our inverse function, something that solves for
    //   the input to the relation we're inverting using only those variables is
    //   the RHS of the inverse. For example, when we solve for t0 the equality
    //   t0=t3 allows us to just return t3---t3 is i which is what the RHS
    //   should have given an the input of j from the LHS.
    //
    //   To get variable names that make sense we use the tuple decl for the
    //   relation. The tuple decl contains all variables in the relation in
    //   order. For our permute relation it would be [i,j,j,i]. T3 indexes into
    //   this and returns "i".
    //
    // For our example relation: the LHS computes to [j,i], to get the RHS we:
    //   (1) ask for a function solving for t0 only using the variables
    //       [t2, t3]. We get: t3.
    //   (2) we use the variables in the tuple decl to print
    //       that variable to a string we get: i.
    //   (3) we repeat (1) and (2) for t1 yielding: j.
    // putting that together with the RHS we get:
    //   {[j,i]->[i,j]}.

    std::string inverseLHS =
        transform->Apply(s0->getIterationSpace())->getTupleDecl().toString();

    std::string inverseRHS;
    TupleDecl decl = transform->getTupleDecl();
    // loop over input to relation we're trying to invert
    for (int inputVar = 0; inputVar < transform->inArity(); inputVar++) {
      // find fuction for variable in input in the output (output is the LHS of
      // function we're inverting, it's the RHS of inverse function).
      Exp *inverseForInputVar = transform->findFunction(
          inputVar, transform->inArity(), transform->arity());

      inverseRHS += inverseForInputVar->prettyPrintString(decl);
      inverseRHS +=
          inputVar == transform->inArity() - 1 ? "" : ", "; // add comma
    }

    std::stringstream ss;
    ss << "affine_map<(" << inverseLHS << ") -> (" << inverseRHS << ")>";

    mlir::AffineMap inverseMap =
        mlir::parseAttribute(ss.str(), rewriter.getContext())
            .cast<AffineMapAttr>()
            .getValue();

    LLVM_DEBUG(llvm::dbgs() << "inverse map: " << inverseMap << "\n");

    std::unordered_map<std::string, mlir::arith::ConstantIndexOp> m;
    m["I"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 2);
    m["K"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 3);
    m["L"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 4);
    m["J"] = rewriter.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 5);
    omega::CG_result *ast = mttkrp.thing();
    auto loop =
        Walker(rewriter, barOp, m, llvm::Optional<mlir::AffineMap>(inverseMap))
            .walk(ast);
    if (!loop) {
      return failure();
    }
    rewriter.eraseOp(barOp);
    LLVM_DEBUG(llvm::dbgs() << "===========================================\n");
    return success();
  }
};
} // end anonymous namespace

void populateStandaloneToSomethingConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ReplaceWithCodeGen>(patterns.getContext());
}

void MyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateStandaloneToSomethingConversionPatterns(patterns);
  ConversionTarget target(getContext());
  target.addLegalDialect<scf::SCFDialect, arith::ArithmeticDialect,
                         vector::VectorDialect, memref::MemRefDialect>();
  target.addIllegalOp<standalone::BarOp>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
