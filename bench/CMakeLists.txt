add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.mlir
                   DEPENDS sparse_mttkrp.mlir
									 COMMAND standalone-opt sparse_mttkrp.mlir
													 -my-pass
													 -inline
													 -cse
													 -lower-affine
													 -convert-vector-to-scf
													 -convert-scf-to-cf
													 -gpu-to-llvm
													 -convert-vector-to-llvm
													 -convert-memref-to-llvm
													 -convert-complex-to-standard
													 -convert-math-to-llvm
													 -convert-complex-to-llvm
													 -convert-math-to-libm
													 -convert-func-to-llvm
													 -reconcile-unrealized-casts
													 -o ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.mlir
                   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.bc
                   DEPENDS ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.mlir
									 COMMAND mlir-translate
													 -mlir-to-llvmir
													 -o ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.bc
                           ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.mlir
                   WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.o
                   DEPENDS ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.bc
									 COMMAND llc
													 -filetype=obj
									         -O3
													 -o ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.o
                           ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.bc
                   WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

add_executable(bench ${PROJECT_BINARY_DIR}/bench/sparse_mttkrp_output.o mttkrp.cpp)
target_link_libraries(bench mlir_c_runner_utils mlir_runner_utils Runtime)
