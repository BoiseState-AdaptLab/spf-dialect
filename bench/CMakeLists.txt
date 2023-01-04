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

	# compile benchmark to mlir llvm dialect
	add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.mlir
										DEPENDS ${mlir_file}
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
														-o ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.mlir
										WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

	# translate from mlir llvm dialect to llvm bytecode
	add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.bc
										DEPENDS ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.mlir
										COMMAND mlir-translate
														-mlir-to-llvmir
														-o ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.bc
														${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.mlir
										WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

	# compile llvm bytecode to object file
	add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.o
										DEPENDS ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.bc
										COMMAND llc
														-filetype=obj
														-O3
														-o ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.o
														${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.bc
										WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

	# append to global list for later use in creating executable
	list(APPEND ${object_file_list} ${PROJECT_BINARY_DIR}/bench/${FILE_NAME_NO_EXTENSION}_output.o)
endmacro()

# create benchmarks ================================================================================
set(MLIR_BENCHMARK_OBJECT_FILES)
create_object_file_and_append(MLIR_BENCHMARK_OBJECT_FILES sparse_mttkrp.mlir)

add_executable(bench ${MLIR_BENCHMARK_OBJECT_FILES} mttkrp.cpp)
target_link_libraries(bench mlir_c_runner_utils mlir_runner_utils Runtime)
