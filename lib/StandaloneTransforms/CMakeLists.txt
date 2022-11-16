add_mlir_library(StandaloneTransforms
  MyPass.cpp
  Printer.h
  Printer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  lib/installed/include/iegenlib
  lib/IEGenLib/lib/installed/include
  lib/IEGenLib/lib/gtest/include

  DEPENDS
  StandaloneTransformsPassIncGen

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
