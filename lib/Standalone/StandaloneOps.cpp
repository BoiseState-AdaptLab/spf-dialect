//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"

mlir::linalg::OpOperandVector mlir::standalone::BarOp::getUFInputOperands() {
  // BarOp has variadic number of input and output parameters. There's a problem
  // with this, how are we to know where to break between operation parameters
  // intended to be fall in the input bucket vs output bucket. The
  // AttrSizedOperandSegments trait solves this problem. The trait requires an
  // operation to have a operand_segment_sizes attribute. When trait is present,
  // tablegen generates `getInputs` and `getOutputs` functions that read the
  // appropriate numer of operands based on this attribute.
  int64_t numExtra = this->getUfInputs().size();
  mlir::linalg::OpOperandVector result;
  result.reserve(numExtra);
  llvm::transform(this->getOperation()->getOpOperands().take_front(numExtra),
                  std::back_inserter(result),
                  [](mlir::OpOperand &opOperand) { return &opOperand; });
  return result;
}

mlir::linalg::OpOperandVector mlir::standalone::BarOp::getInputOperands() {
  int64_t numInputs = this->getInputs().size();
  // Regular inputs come after uf inputs. To clarify the difference, a UF input
  // may be used as an argument to a UF (Uninterpreted Function) but will not be
  // used to generate a load used in the statement.
  int64_t numUFInputs = this->getUfInputs().size();
  mlir::linalg::OpOperandVector result;
  // TODO: this is copy pasted from the equivalent linalg op. I don't know what
  // the point of this transform thing is. I think it might just be an obnoxious
  // way to do a for loop and push back onto result.
  result.reserve(numInputs);
  llvm::transform(this->getOperation()
                      ->getOpOperands()
                      .drop_front(numUFInputs)
                      .take_front(numInputs),
                  std::back_inserter(result),
                  [](mlir::OpOperand &opOperand) { return &opOperand; });
  return result;
}

mlir::linalg::OpOperandVector mlir::standalone::BarOp::getOutputOperands() {
  int64_t numOutputs = this->getOutputs().size();
  mlir::linalg::OpOperandVector result;
  result.reserve(numOutputs);
  llvm::transform(this->getOperation()->getOpOperands().take_back(numOutputs),
                  std::back_inserter(result),
                  [](OpOperand &opOperand) { return &opOperand; });
  return result;
}