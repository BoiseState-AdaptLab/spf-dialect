#ifndef SPF_DIALECT_PASSES_H
#define SPF_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include "SPF/SPFOps.h"
#include <limits>

namespace mlir {
    namespace spf {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

        std::unique_ptr<OperationPass<>> createConvertSPFToLoops();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION

#include "SPFTransforms/Passes.h.inc"

    }
}

#endif //SPF_DIALECT_PASSES_H
