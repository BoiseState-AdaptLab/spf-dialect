# helpers ==========================================================================================
# sets up commands to create object file from MLIR file and appends to
# MLIR_BENCHMARK_OBJECT_FILES list
macro(create_object_file_and_append object_file_list mlir_file)
  # Turn the filename into list [filename_no_extension, extension]
  # https://stackoverflow.com/a/5272993/3217397
  string(REPLACE "." ";" FILE_PARTS ${mlir_file})
  list(LENGTH FILE_PARTS list_len)

  # Grab the fist thing from the list one
  # https://stackoverflow.com/a/56586927/3217397
  list(GET FILE_PARTS 0 FILE_NAME_NO_EXTENSION)

	# lower benchmark out of mlir spf dialect
	set(PARTIAL_LOWERING_FILENAME ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_partial_lowering.mlir)
	add_custom_command(OUTPUT ${PARTIAL_LOWERING_FILENAME}
										DEPENDS ${mlir_file}
										COMMAND standalone-opt ${mlir_file}
														-my-pass
														-inline
														-o ${PARTIAL_LOWERING_FILENAME}
										WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

	# optimize benchmark and lower to mlir llvm dialect
	set(FULL_LOWERING_FILENAME ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.mlir)
	add_custom_command(OUTPUT ${FULL_LOWERING_FILENAME}
										DEPENDS ${PARTIAL_LOWERING_FILENAME}
										COMMAND standalone-opt ${PARTIAL_LOWERING_FILENAME}
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
														-o ${FULL_LOWERING_FILENAME}
										WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

	# translate from mlir llvm dialect to llvm bytecode
	set(BYTECODE_FILENAME ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.bc)
	add_custom_command(OUTPUT ${BYTECODE_FILENAME}
										DEPENDS ${FULL_LOWERING_FILENAME}
										COMMAND mlir-translate
														-mlir-to-llvmir
														-o ${BYTECODE_FILENAME}
														${FULL_LOWERING_FILENAME}
										WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

	# compile llvm bytecode to object file
	set(OBJECT_FILE_FILENAME ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.o)
	add_custom_command(OUTPUT ${OBJECT_FILE_FILENAME}
										DEPENDS ${BYTECODE_FILENAME}
										COMMAND llc
														-filetype=obj
														-O3
														-o ${OBJECT_FILE_FILENAME}
														${BYTECODE_FILENAME}
										WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

	# append to global list for later use in creating executable
	list(APPEND ${object_file_list} ${OBJECT_FILE_FILENAME})
endmacro()

# create benchmarks ================================================================================
set(MLIR_BENCHMARK_OBJECT_FILES)
create_object_file_and_append(MLIR_BENCHMARK_OBJECT_FILES sparse_mttkrp.mlir)
create_object_file_and_append(MLIR_BENCHMARK_OBJECT_FILES sparse_mttkrp_extra_stuff.mlir)

add_executable(bench ${MLIR_BENCHMARK_OBJECT_FILES} mttkrp.cpp)
target_link_libraries(bench
											mlir_c_runner_utils
											mlir_runner_utils
											Runtime
											LLVMSupport
											)

target_compile_options(bench PUBLIC "-O3")