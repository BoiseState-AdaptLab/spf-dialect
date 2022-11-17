//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "mlir/IR/OpBase.td"
include "Standalone/StandaloneDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_YieldOp : Standalone_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "TODO";
  let description = [{
    TODO
  }];
}

def Standalone_BarOp : Standalone_Op<"bar", [AttrSizedOperandSegments]> {
    let summary = "This is a bar op";
    let description = [{
        Stuff
    }];

    let arguments = (ins    Variadic<AnyType>:$inputs,
                            Variadic<AnyShaped>:$outputs,
                            ArrayAttr:$ufNames,
                            AffineMapArrayAttr:$reads,
                            AffineMapArrayAttr:$writes);

    let extraClassDeclaration = [{
        mlir::linalg::OpOperandVector getInputOperands();
        mlir::linalg::OpOperandVector getOutputOperands();
    }];

    let regions = (region AnyRegion:$body, VariadicRegion<AnyRegion>:$ufs);
}

#endif // STANDALONE_OPS
