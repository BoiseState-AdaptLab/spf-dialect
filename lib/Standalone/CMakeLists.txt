add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SPF

        DEPENDS
        MLIRSPFOpsIncGen

		LINK_LIBS PUBLIC
		MLIRIR
        MLIRInferTypeOpInterface
	)
