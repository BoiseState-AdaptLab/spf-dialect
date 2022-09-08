#ifndef STANDALONE_DIALECT_PASSES_H
#define STANDALONE_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include "Standalone/StandaloneOps.h"
#include <limits>

namespace mlir {
    namespace standalone {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

        std::unique_ptr<OperationPass<>> createMyPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION

#include "StandaloneTransforms/Passes.h.inc"

    }
}

#endif //STANDALONE_DIALECT_PASSES_H
