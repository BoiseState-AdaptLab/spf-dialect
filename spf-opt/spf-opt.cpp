#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "SPF/SPFDialect.h"
#include "SPFTransforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::spf::registerConvertSPFToLoops();

  mlir::DialectRegistry registry;
  registry.insert<mlir::spf::SPFDialect>();
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SPF optimizer driver\n", registry));
}
