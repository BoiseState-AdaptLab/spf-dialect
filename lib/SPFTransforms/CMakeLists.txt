add_mlir_library(SPFTransforms
  MyPass.cpp
  Printer.h
  Utils.h

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  lib/installed/include/iegenlib
  lib/IEGenLib/lib/installed/include
  lib/IEGenLib/lib/gtest/include

  DEPENDS
  SPFTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRLoopLikeInterface
  MLIRPass
  MLIRSupport
  MLIRTransformUtils

  PRIVATE
  iegenlib
  gmp
  isl
  codegen
  omega
  )
