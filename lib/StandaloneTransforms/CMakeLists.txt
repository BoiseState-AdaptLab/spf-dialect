add_mlir_library(StandaloneTransforms
  MyPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  StandaloneTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRLoopLikeInterface
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )