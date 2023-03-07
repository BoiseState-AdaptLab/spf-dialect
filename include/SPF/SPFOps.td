#ifndef SPF_OPS
#define SPF_OPS

include "mlir/IR/OpBase.td"
include "SPF/SPFDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def SPF_YieldOp : SPF_Op<"yield", [Pure, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "yield for SPF statements";
  let description = [{
    Results that are provided to yield will be written to the inputs of an
    spf.statement operation with the mapping provided in writes.
  }];
}

def SPF_StatementOp : SPF_Op<"statement", [AttrSizedOperandSegments]> {
    let summary = "SPF statement";
    let description = [{
        See Chapter 3 of thesis: "Portable Sparse Polyhedral Framework code
        generation using Multi Level Intermediate Representation" by Aaron St
        George at Boise State University for line by line description of
        statement operation.
    }];

    let arguments = (ins Variadic<AnyType>:$symbols,
                         Variadic<AnyType>:$ufInputs,
                         Variadic<AnyType>:$inputs,
                         Variadic<AnyType>:$outputs,
                         ArrayAttr:$symbolNames,
                         ArrayAttr:$iteratorTypes,
                         StrAttr:$iterationSpace,
                         StrAttr:$executionSchedule,
                         DefaultValuedAttr<BoolAttr, "false">:$atomicWrite,
                         ArrayAttr:$reads,
                         ArrayAttr:$writes,
                         ArrayAttr:$transforms);

    let extraClassDeclaration = [{
        mlir::OpOperandVector getSymbolOperands();
        mlir::OpOperandVector getUFInputOperands();
        mlir::OpOperandVector getInputOperands();
        mlir::OpOperandVector getOutputOperands();
    }];

    let regions = (region AnyRegion:$body, VariadicRegion<AnyRegion>:$ufs);
}

def SPF_ComputationOp : SPF_Op<"computation", [NoTerminator]> {
    let summary = "SPF computation";
    let description = [{
        All statement operations must be inside a computation. After lowering,
        generated code will execute in the order determined by execution
        schedule, not by order within computation.
    }];

    let regions = (region AnyRegion:$body);
}

#endif // SPF_OPS
