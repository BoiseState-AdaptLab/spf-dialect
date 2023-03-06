#ifndef SPF_DIALECT
#define SPF_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SPF dialect definition.
//===----------------------------------------------------------------------===//

def SPF_Dialect : Dialect {
    let name = "spf";
    let summary = "A dialect for the Sparse Polyhedral Framework.";
    let description = [{
        This dialect wraps existing SPF (Sparse Polyhedral Framework) tools to
        provide an MLIR interface and code generation. Generating MLIR allows
        SPF based tools to target a much broader set of hardware than previously
        possible, and allows researchers to use SPF to create optimizations
        inside MLIR based compilers.
    }];
    let cppNamespace = "::mlir::spf";

    let dependentDialects = ["scf::SCFDialect", "vector::VectorDialect", "mlir::AffineDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// Base SPF operation definition.
//===----------------------------------------------------------------------===//

class SPF_Op<string mnemonic, list<Trait> traits = []> :
        Op<SPF_Dialect, mnemonic, traits>;

#endif // SPF_DIALECT
