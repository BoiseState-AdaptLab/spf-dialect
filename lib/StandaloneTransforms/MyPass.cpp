#include "Parser.h"
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
#include "mlir/IR/MLIRContext.h"
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

#include <cassert>
#include <code_gen/CG.h>
#include <code_gen/codegen_error.h>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <omega.h>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define DEBUG_TYPE "my-pass"

using ReadWrite = std::vector<std::pair<std::string, std::string>>;

namespace sparser = mlir::standalone::parser;

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

struct Walker : public sparser::VisitorBase {
  mlir::OpBuilder &builder;
  std::unordered_map<std::string, mlir::Value> symbols;
  std::unordered_map<std::string, mlir::Region *> ufNameToRegion;

  standalone::ComputationOp computationOp;
  std::vector<StatementContext> statements;
  std::unordered_map<std::string, mlir::Value> ivs; // induction variables
  int loopLevel = 0;

public:
  explicit Walker(mlir::OpBuilder &builder,
                  standalone::ComputationOp computationOp,
                  std::vector<StatementContext> statements)
      : builder(builder), computationOp(computationOp), statements(statements) {

    // populate ufNameToRegion TODO: ufs should be functions stored in the
    // symbol table. They definitely shouldn't be read off the first statement
    // like this. This is bad.
    auto ufs = statements[0].statementOp.getUfs();
    auto ufNames = statements[0].statementOp.getUfNames();
    for (auto &attr : llvm::enumerate(ufNames)) {
      std::string ufName = attr.value().dyn_cast_or_null<StringAttr>().str();
      // This *might* be sketchy... I think all regions are stored in the main
      // context and not moved around, in which case this is safe. But I might
      // be wrong. Asan hasn't told me about any issues yet.
      ufNameToRegion[ufName] = &ufs[attr.index()];
    }

    // populate symbols TODO: symbols should hang off computation not statement,
    // for now I'm just assuming that the symbols are the same and reading them
    // off the first op. This is definitely not a reasonable thing to do.
    SmallVector<Value> symbolOperands =
        statements[0].statementOp.getSymbolOperands();
    auto symbolNames = statements[0].statementOp.getSymbolNames();
    for (auto &attr : llvm::enumerate(symbolNames)) {
      std::string symbolName =
          attr.value().dyn_cast_or_null<StringAttr>().str();
      symbols[symbolName] = symbolOperands[attr.index()];
    }
  }

  // This function is very useful for debugging purposes
  void dumpItAll() __attribute__((noinline, used)) {
    LLVM_DEBUG({
      // Find the top-level operation.
      auto *topLevelOp = computationOp.getOperation();
      while (auto *parentOp = topLevelOp->getParentOp()) {
        topLevelOp = parentOp;
      }
      topLevelOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    });
  }

  void codeGen(std::unique_ptr<sparser::Program> p) {
    LLVM_DEBUG(llvm::dbgs() << "Walker ====================================\n");
    for (auto &statement : p->statements) {
      statement->accept(*this);
    }
  }

private:
  void visit(sparser::LoopAST *loop) override {
    LLVM_DEBUG(llvm::dbgs() << "loop"
                            << "\n");
    auto stopString = loop->stop->symbol;
    if (stopString.empty() || symbols.find(stopString) == symbols.end()) {
      std::cerr << "err: "
                << "oh no!" << std::endl;
      exit(1);
    }

    auto start = builder.create<mlir::arith::ConstantIndexOp>(
        computationOp.getLoc(), loop->start);

    // Create mutable AffineMap for applying increment to stop symbol
    auto map = MutableAffineMap(
        AffineMap::getMultiDimIdentityMap(1, builder.getContext()));

    // Update map output to include increment
    auto result = map.getResult(0);
    map.setResult(0, result + loop->stop->increment);

    assert(map.getNumResults() == 1 && "map should return ");

    // create stop by applying the map
    auto stop = makeCanonicalAffineApplies(builder, computationOp->getLoc(),
                                           map.getAffineMap(),
                                           {symbols[stopString]})[0];

    auto step = builder.create<mlir::arith::ConstantIndexOp>(
        computationOp.getLoc(), loop->step);

    scf::ForOp forOp =
        builder.create<scf::ForOp>(computationOp.getLoc(), start, stop, step);

    assert(ivs.find(loop->inductionVar) == symbols.end() &&
           "an induction variable should never be reused in new loop");
    // store off induction variable
    ivs[loop->inductionVar] = forOp.getInductionVar();

    // generate code for body of loop
    builder.setInsertionPointToStart(forOp.getBody());
    for (auto &statement : loop->block) {
      statement->accept(*this);
    }

    // As we're not inside the loop anymore, this isn't a valid induction
    // variable.
    ivs.erase(loop->inductionVar);

    // reset insertion point for next statement
    builder.setInsertionPointAfter(forOp);
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
  void visit(sparser::StatementCallAST *call) override {
    LLVM_DEBUG(llvm::dbgs() << "call"
                            << "\n");

    // Here we build up MLIR arguments to the statement by finding the
    // corresponding generated MLIR value for each argument in the Omega AST.
    std::vector<mlir::Value> args;
    for (auto &arg : call->args) {
      llvm::TypeSwitch<sparser::SymbolOrInt *>(arg.get())
          .Case<sparser::Symbol>([&](sparser::Symbol *symbol) {
            assert(symbol->increment == 0 &&
                   "don't know what to do with increment in statement call");

            bool isIV = ivs.find(symbol->symbol) != ivs.end();
            bool isSymbol = symbols.find(symbol->symbol) != symbols.end();

            assert(!symbol->symbol.empty() && (isIV || isSymbol) &&
                   "can't find induction variable for statement call variable");

            assert(!(isIV && isSymbol) && "no idea why this would ever happen");

            if (isIV) {
              args.push_back(ivs[symbol->symbol]);
            } else {
              args.push_back(symbols[symbol->symbol]);
            }
          })
          .Case<sparser::Int>([&](sparser::Int *integer) {
            args.push_back(builder.create<mlir::arith::ConstantIndexOp>(
                computationOp.getLoc(), integer->val));
          })
          .Default([&](sparser::SymbolOrInt *symbolOrInt) {
            LLVM_DEBUG(llvm::errs() << "unknown SymbolOrInt,kind<"
                                    << symbolOrInt->getKind() << ">\n");
            exit(1);
          });
    }

    auto loc = computationOp->getLoc();

    int statementIndex = call->statementIndex;

    // get the statement being called
    auto statementOp = statements[statementIndex].statementOp;

    // The statement op provides data access functions: read and write maps
    // written with an iteration space tuple as input and where to read out of a
    // memref as output. Omega generates statement calls with the execution
    // schedule tuple as input. These two things won't work together... But, if
    // we can compose a function from (possibly transformed) execution schedule
    // back to iteration space (aka the composition of the inverse of all
    // transformations done) with the data access functions then we have a map
    // from omega generated statement calls to where to read or write out of
    // memref.
    auto inverse =
        statements[statementIndex].getExecutionScheduleToIterationSpace();
    auto readMaps = llvm::map_range(
        statementOp.getReads().getAsValueRange<AffineMapAttr>(),
        [&](AffineMap map) -> AffineMap { return map.compose(inverse); });
    auto writeMaps = llvm::map_range(
        statementOp.getWrites().getAsValueRange<AffineMapAttr>(),
        [&](AffineMap map) -> AffineMap { return map.compose(inverse); });

    SmallVector<Value> indexedValues;
    indexedValues.reserve(statementOp.getInputs().size());
    { // 1.a. produce loads from input memrefs
      SmallVector<Value> inputOperands = statementOp.getInputOperands();
      for (size_t i = 0; i < inputOperands.size(); i++) {
        // read the map that corresponds with the current inputOperand. It seems
        // like this would be better using the subscript "[]" operator, but
        // indexingMaps doesn't provide one.
        auto map = *(readMaps.begin() + i);
        auto indexing = makeCanonicalAffineApplies(builder, loc, map, args);
        indexedValues.push_back(
            builder.create<memref::LoadOp>(loc, inputOperands[i], indexing));
      }
    }

    // 1.b. Emit load for output memrefs
    SmallVector<Value> outputOperands = statementOp.getOutputOperands();
    for (size_t i = 0; i < outputOperands.size(); i++) {
      // read the map that corresponds with the current inputOperand.
      AffineMap map = *(writeMaps.begin() + i);

      SmallVector<Value> indexing =
          makeCanonicalAffineApplies(builder, loc, map, args);
      indexedValues.push_back(
          builder.create<memref::LoadOp>(loc, outputOperands[i], indexing));
    }

    // TODO: there's no validation that the region has a block.
    auto &block = statementOp.getBody().front();

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
      AffineMap map = *(writeMaps.begin() + i);
      indexing.push_back(makeCanonicalAffineApplies(builder, loc, map, args));
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

  void visit(sparser::UFAssignmentAST *ufAssignment) override {
    LLVM_DEBUG(llvm::dbgs() << "uf"
                            << "\n");

    // Find region associated with UF (Uninterpreted Function)
    std::string ufName = ufAssignment->ufName;
    if (ufNameToRegion.find(ufName) == ufNameToRegion.end()) {
      std::cerr << "err: could not find uninterpreted function" << std::endl;
      exit(1);
    }

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
    //  - inputs: loads are generated from inputs that are used as arguments to
    //    the statement.
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
    // ^bb0(%uf_b_coord_0,: memref<?xindex>, // This needs to be mapped from
    //                                       // b_coord_0 UF input.
    //      %uf_b_coord_1,: memref<?xindex>, // This needs to be mapped from
    //                                       // b_coord_1 UF input.
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
    SmallVector<Value> ufArgs = statements[0].statementOp.getUFInputOperands();

    // Here we build up MLIR arguments to the UF by finding the corresponding
    // generated MLIR value for each argument in the Omega AST.
    for (auto &arg : ufAssignment->args) {
      llvm::TypeSwitch<sparser::SymbolOrInt *>(arg.get())
          .Case<sparser::Symbol>([&](sparser::Symbol *symbol) {
            assert(symbol->increment == 0 &&
                   "don't know what to do with increment in statement call");
            assert(!symbol->symbol.empty() &&
                   ivs.find(symbol->symbol) != ivs.end() &&
                   "can't find induction variable for statement call variable");

            ufArgs.push_back(ivs[symbol->symbol]);
          })
          .Case<sparser::Int>([&](sparser::Int *integer) {
            ufArgs.push_back(builder.create<mlir::arith::ConstantIndexOp>(
                computationOp.getLoc(), integer->val));
          })
          .Default([&](sparser::SymbolOrInt *symbolOrInt) {
            LLVM_DEBUG(llvm::errs() << "unknown SymbolOrInt,kind<"
                                    << symbolOrInt->getKind() << ">\n");
            exit(1);
          });
    }

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
    ivs[ufAssignment->inductionVar] = iv;
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

std::unique_ptr<standalone::parser::Program> parse(std::string s) {
  auto lexer = standalone::parser::Lexer(std::move(s));
  auto parser = standalone::parser::Parser(lexer);
  return parser.parseProgram();
}

class ReplaceWithCodeGen : public OpRewritePattern<standalone::ComputationOp> {
public:
  using OpRewritePattern<standalone::ComputationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::ComputationOp computationOp,
                                PatternRewriter &rewriter) const override {
    Computation computation; // IEGenLib computation (MLIR computationOp is
                             // directly analogous)

    int statementIndex = 0;
    std::vector<StatementContext> statements;
    // Run through MLIR statements in MLIR computationOp and populate IEGenLib
    // computation.
    for (auto &op : computationOp.getBody().front()) {
      if (!isa<standalone::BarOp>(op)) {
        return emitError(op.getLoc(),
                         "A computation can only contain statements");
      }
      standalone::BarOp statement = cast<standalone::BarOp>(op);

      // Build up reads and writes
      ReadWrite reads;
      ReadWrite writes;
      { // create reads for inputs
        auto readMaps = statement.getReads().getAsValueRange<AffineMapAttr>();
        for (size_t i = 0; i < statement.getInputOperands().size(); i++) {
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
        auto writeMaps = statement.getWrites().getAsValueRange<AffineMapAttr>();
        for (size_t i = 0; i < statement.getOutputOperands().size(); i++) {
          AffineMap map = *(writeMaps.begin() + i);
          // The += operator counts as both read and write.
          std::string read_write = relationForOperand(map);

          char name[100];
          std::sprintf(name, "output_%zu", i);
          reads.push_back({name, read_write});
          writes.push_back({name, read_write});
        }
      }

      // create IEGenLib statement and add to IEGenLib computation
      // NOTE: intentially leaking s here. Lots of stuff inside IEGenLib doesn't
      // handle memory properly and I don't have time to fix all that.
      Stmt *s = new Stmt("", statement.getIterationSpace().str(),
                         statement.getExecutionSchedule().str(), reads, writes);
      computation.addStmt(s);

      auto statementContext =
          StatementContext(rewriter.getContext(), statement,
                           s->getIterationSpace(), s->getExecutionSchedule());

      // Add any transformations
      for (auto &attr : statement.getTransforms()) {
        std::string transform = attr.dyn_cast_or_null<StringAttr>().str();
        Relation *relation = new Relation(transform);
        computation.addTransformation(statementIndex, relation);
        statementContext.addTransformation(relation);
        // NOTE: intentionally leaking relation
      }

      statements.push_back(std::move(statementContext));

      statementIndex++;
    }

    LLVM_DEBUG(llvm::dbgs() << "IEGenLib codeGen ==========================\n");
    LLVM_DEBUG(llvm::dbgs() << computation.codeGen());

    LLVM_DEBUG(llvm::dbgs() << "codeJen ===================================\n");
    std::string codeJen = computation.codeJen();
    LLVM_DEBUG(llvm::dbgs() << codeJen);

    LLVM_DEBUG(llvm::dbgs() << "parse =====================================\n");
    auto simpleAST = parse(codeJen);
    if (simpleAST) {
      LLVM_DEBUG(llvm::dbgs() << simpleAST->dump() << "\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "mapping functions =========================\n");
    for (auto &statement : llvm::enumerate(statements)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "statement: " << statement.index()
                 << ", executionScheduleToIterationSpace(AffineMap): "
                 << statement.value().getExecutionScheduleToIterationSpace()
                 << "\n");
    }

    // Walk ast and generate MLIR based on Omega AST
    Walker(rewriter, computationOp, statements).codeGen(std::move(simpleAST));

    rewriter.eraseOp(computationOp);
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
                         vector::VectorDialect, memref::MemRefDialect,
                         AffineDialect>();
  target.addIllegalOp<standalone::BarOp, standalone::ComputationOp>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
