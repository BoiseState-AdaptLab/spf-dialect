#include "Printer.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "StandaloneTransforms/Passes.h"
#include "Utils.h"
#include "iegenlib.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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
  explicit Walker(mlir::OpBuilder &builder, standalone::BarOp barOp,
                  llvm::Optional<mlir::AffineMap> inverseMap =
                      llvm::Optional<mlir::AffineMap>())
      : builder(builder), barOp(barOp), inverseMap(inverseMap) {
    // populate ufNameToRegion
    auto ufs = barOp.getUfs();
    auto ufNames = barOp.getUfNames();
    for (auto &attr : llvm::enumerate(ufNames)) {
      std::string ufName = attr.value().dyn_cast_or_null<StringAttr>().str();
      // This *might* be sketchy... I think all regions are stored in the main
      // context and not moved around, in which case this is safe. But I might
      // be wrong. Asan hasn't told me about any issues yet.
      ufNameToRegion[ufName] = &ufs[attr.index()];
    }

    // populate symbols
    SmallVector<Value> symbolOperands = barOp.getSymbolOperands();
    auto symbolNames = barOp.getSymbolNames();
    for (auto &attr : llvm::enumerate(symbolNames)) {
      std::string symbolName =
          attr.value().dyn_cast_or_null<StringAttr>().str();
      symbols[symbolName] = symbolOperands[attr.index()];
    }

    // For aesthetic reasons it is nice to only have one zero and 1 constant op
    // created.
    zero = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 0);
    one = builder.create<mlir::arith::ConstantIndexOp>(barOp.getLoc(), 1);
  }

  llvm::Optional<mlir::Operation *> walk(omega::CG_result *t) {
    LLVM_DEBUG(llvm::dbgs() << "Walker ====================================\n");
    LLVM_DEBUG(llvm::dbgs() << "result\n");
    dispatch(t);
    return maybeOp;
  }

  mlir::OpBuilder &builder;
  std::unordered_map<std::string, mlir::Value> symbols;
  std::unordered_map<std::string, mlir::Region *> ufNameToRegion;
  mlir::Value zero;
  mlir::Value one;
  llvm::Optional<mlir::Operation *> maybeOp;
  standalone::BarOp barOp;
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
                << "unknown omega::CG_result type" << std::endl;
      exit(1);
    }
  }

  void walkLoop(omega::CG_loop *loop) {
    LLVM_DEBUG(llvm::dbgs() << "loop[");
    LLVM_DEBUG(llvm::dbgs() << " level:" << loop->level_);
    LLVM_DEBUG(llvm::dbgs() << " need:" << (loop->needLoop_ ? "y" : "n"));

    auto bounds = const_cast<omega::Relation &>(loop->bounds_);

    // Loops will be created for each level in the execution schedule. Some
    // levels will require a loop to be generated, some a call to an
    // uninterpreted function, some don't require any code to be generated.
    if (loop->needLoop_) {
      // (Should be) set while looping over greater than or equal to conjuncts.
      std::string upperBound;

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
              upperBound = var.curr_var()->name();
              LLVM_DEBUG(llvm::dbgs() << " over:" << var.curr_var()->name());
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

      if (upperBound.empty() || symbols.find(upperBound) == symbols.end()) {
        std::cerr << "err: "
                  << "oh no!" << std::endl;
        exit(1);
      }

      auto upperBoundValue = symbols[upperBound];

      // omega 1 indexes "loop->level_", hence the -1
      auto iteratorType = barOp.getIteratorTypes()[loop->level_ - 1]
                              .dyn_cast_or_null<StringAttr>();

      // Generate loops.
      //
      // TODO: it would be nice to template this but there doesn't appear to be
      // a common API between ParallelOp and ForOp. There is a LoopLike
      // Interface but it seems like it doesn't ahve everything we need. Maybe
      // what we need could be added upstream?
      if (iteratorType &&
          iteratorType.getValue() ==
              "parallel") { // add parallel loop, and update walker state
        scf::ParallelOp parallelOp = builder.create<scf::ParallelOp>(
            barOp.getLoc(), mlir::ValueRange(zero),
            mlir::ValueRange(upperBoundValue), mlir::ValueRange(one));

        // if this is the top level loop store it off to return
        if (!maybeOp) {
          maybeOp = parallelOp;
        }

        // store off induction variable
        ivs.push_back(parallelOp.getInductionVars().front());

        // Start add future loops inside this loop
        builder.setInsertionPointToStart(parallelOp.getBody());
      } else { // add regular for loop, and update walker state
        scf::ForOp forOp = builder.create<scf::ForOp>(barOp.getLoc(), zero,
                                                      upperBoundValue, one);

        // if this is the top level loop store it off to return
        if (!maybeOp) {
          maybeOp = forOp;
        }

        // store off induction variable
        ivs.push_back(forOp.getInductionVar());

        // Start add future loops inside this loop
        builder.setInsertionPointToStart(forOp.getBody());
      }
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

            // At this point we have a UF call and we need to grab the
            // associated region from the operation and inline it.

            // Find region associated with UF (Uninterpreted Function)
            std::string ufName = unmangleUfName(var.curr_var()->name());
            if (ufNameToRegion.find(ufName) == ufNameToRegion.end()) {
              std::cerr << "err: could not find uninterpreted function"
                        << std::endl;
              exit(1);
            }
            LLVM_DEBUG(llvm::dbgs() << " uf_call:" << ufName);

            // TODO: why aren't blocks value types?
            // Most MLIR types (Attribute for example) are lightweight wrappers
            // around pointers to storage in the main context that (I think) is
            // guaranteed not to move around. What makes blocks different?
            auto &ufBlock = ufNameToRegion[ufName]->front();

            // TODO: this should probably be done with symbols:
            // https://mlir.llvm.org/docs/SymbolsAndSymbolTables/ inside
            // something like integer sets:
            // https://mlir.llvm.org/docs/Dialects/Affine/#integer-sets that
            // would replace the string execution schedule.
            //
            // We need to create a mapping from the UF arguments to the
            // generated induction variables and UF inputs to the operation. An
            // example is given below:
            //
            // Inputs to the operation are in 3 segments: UF inputs, inputs, and
            // outputs.
            //  - UF inputs: used in UFs but not in the statement.
            //  - inputs: loads are generated from inputs that are used as
            //    arguments to the statement.
            //  - output: loads and stores are generated for/from the statement.
            //
            // // a += b * c  where b is a sparse matrix and c is dense
            // "standalone.bar"(
            //                   %b_coord_0, // UF input
            //                   %b_coord_1, // UF input
            //                   %b_values,  // input
            //                   %c,         // input
            //                   %a          // output
            //                 ) ({ //this region represents be the statement
            // // statement region omitted.
            // }, { // this region is the first UF
            // ^bb0(%uf_b_coord_0,: memref<?xindex>, // This needs to be
            //                                       // mapped from
            //                                       // b_coord_0 UF
            //                                       // input.
            //      %uf_b_coord_1,: memref<?xindex>, // This needs to be
            //                                       // mapped from
            //                                       // b_coord_1 UF
            //                                       // input.
            //      %z index // This needs to be mapped from generated induction
            //               // variable ranging from 0 to NNZ.
            //     ) :
            //  // This region is UF 0, it's expected to have the same arguments
            //  // as the operation's UF input section but can also take any
            //  // induction variables generated by the time it's called.
            // %i = memref.load %uf_argb_coord_0[%z] : memref<?xindex>
            // // This UF generates %i induction variable.
            // "standalone.yield"(%i) : (index) -> ()
            // }, { // this would be another UF
            // // omitted.
            // })  { // attribues
            //       // operand_segment_sizes indicates how large the UF input,
            //       // input, and output segments are.
            //       operand_segment_sizes = dense<[2,2, 1]> : vector<3xi32>,
            //
            //       // other attributes omitted.
            //     } : (memref<?xindex>, memref<?xindex>, memref<?xf64>,
            //          memref<?x?xf64>, memref<?x?xf64>) -> ()
            //
            // The UF in the example needs a mapping from the operation's 2 UF
            // inputs (%b_coord_0, %b_coord_1) and the generated induction
            // variable to the arguments to the UF (%uf_b_coord_0, uf_b_coord_1,
            // %z).
            SmallVector<Value> ufArgs = barOp.getUFInputOperands();
            ufArgs.insert(ufArgs.end(), ivs.begin(), ivs.end());
            BlockAndValueMapping map; // holds a mapping between values.
            map.map(/*from*/ ufBlock.getArguments(),
                    /*to*/ ufArgs);

            // Inlines the UF region with the mapping we created above.
            for (auto &op : ufBlock.without_terminator()) {
              // clone creates the op with the map applied.
              builder.clone(op, map);
            }

            // The UF generates an induction variable with the terminator.
            Operation *terminator = ufBlock.getTerminator();
            OpOperand &operand = terminator->getOpOperands()[0];
            Value iv = map.lookupOrDefault(operand.get());
            ivs.push_back(iv);
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << " ]\n");
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
    indexedValues.reserve(barOp.getInputs().size());
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

    // Build IEGenLib representation from MLIR operation
    Computation computation;
    Stmt *s0 = new Stmt("", barOp.getIterationSpace().str(),
                        barOp.getExecutionSchedule().str(), reads, writes);

    computation.addStmt(s0);

    LLVM_DEBUG(llvm::dbgs() << "IEGenLib codeGen ==========================\n");
    LLVM_DEBUG(llvm::dbgs() << computation.codeGen());

    // LLVM_DEBUG(llvm::dbgs() << "Adding fake transformation ================\n");
    // LLVM_DEBUG(llvm::dbgs() << "transform: {[i,k,l,j] -> [k,i,l,j]}"
    //                         << "\n");
    // auto transform = new Relation("{[i,k,l,j] -> [k,i,l,j]}");
    // mttkrp.addTransformation(0, transform);
    // AffineMap inverseMap = createInverse(transform, s0, rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "inverse map: " << inverseMap << "\n");

    // generate MLIR from omega AST
    omega::CG_result *ast = computation.thing();
    auto loop = Walker(rewriter, barOp).walk(ast);

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
