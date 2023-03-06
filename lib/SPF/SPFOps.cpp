#include "SPF/SPFOps.h"
#include "SPF/SPFDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "SPF/SPFOps.cpp.inc"

// Symbols are used in the iteration space and mapped to arguments using the
// symbolNames attribute.
//
// TODO: I think this way of doing it is pretty greasy. But I need to get this
// done, and it's expedient, so OK for now at least. I think the iteration space
// shouldn't be passed as a string, it should become a first class MLIR concept
// similar to integer sets:
// https://mlir.llvm.org/docs/Dialects/Affine/#integer-sets and then symbols
// could be passed the normal way they are with that construct. But, that's
// going to have to be future work.
mlir::OpOperandVector mlir::spf::BarOp::getSymbolOperands() {
  // BarOp has variadic number of input and output parameters. There's a problem
  // with this, how are we to know where to break between operation parameters
  // intended to be fall in the input bucket vs output bucket. The
  // AttrSizedOperandSegments trait solves this problem. The trait requires an
  // operation to have a operand_segment_sizes attribute. When trait is present,
  // tablegen generates `getInputs` and `getOutputs` functions that read the
  // appropriate numer of operands based on this attribute.
  int64_t numSymbols = this->getSymbols().size();
  mlir::OpOperandVector result;
  result.reserve(numSymbols);
  llvm::transform(this->getOperation()->getOpOperands().take_front(numSymbols),
                  std::back_inserter(result),
                  [](mlir::OpOperand &opOperand) { return &opOperand; });
  return result;
}

// UFIntputOperands are used as arguments to the provided UFs but not as
// arguments to the statement/.
mlir::OpOperandVector mlir::spf::BarOp::getUFInputOperands() {
  // UF inputs come after symbols in the operand list, make sure to skip those.
  int64_t numSymbols = this->getSymbols().size();

  int64_t numUFInputs = this->getUfInputs().size();
  mlir::OpOperandVector result;
  result.reserve(numUFInputs);
  llvm::transform(this->getOperation()
                      ->getOpOperands()
                      .drop_front(numSymbols)
                      .take_front(numUFInputs),
                  std::back_inserter(result),
                  [](mlir::OpOperand &opOperand) { return &opOperand; });
  return result;
}

// Items read from inputs will be arguments to the statement.
mlir::OpOperandVector mlir::spf::BarOp::getInputOperands() {
  int64_t numInputs = this->getInputs().size();
  // Regular inputs come after uf inputs and symbols in the operand list, make
  // sure to skip those.
  int64_t numSymbols = this->getSymbols().size();
  int64_t numUFInputs = this->getUfInputs().size();
  int64_t toSkip = numSymbols + numUFInputs;
  mlir::OpOperandVector result;
  // TODO: this is copy pasted from the equivalent linalg op. I don't know what
  // the point of this transform thing is. I think it might just be an obnoxious
  // way to do a for loop and push back onto result.
  result.reserve(numInputs);
  llvm::transform(
      this->getOperation()->getOpOperands().drop_front(toSkip).take_front(
          numInputs),
      std::back_inserter(result),
      [](mlir::OpOperand &opOperand) { return &opOperand; });
  return result;
}

// Each execution of a statement will write to something in the output
mlir::OpOperandVector mlir::spf::BarOp::getOutputOperands() {
  int64_t numOutputs = this->getOutputs().size();
  mlir::OpOperandVector result;
  result.reserve(numOutputs);
  llvm::transform(this->getOperation()->getOpOperands().take_back(numOutputs),
                  std::back_inserter(result),
                  [](OpOperand &opOperand) { return &opOperand; });
  return result;
}