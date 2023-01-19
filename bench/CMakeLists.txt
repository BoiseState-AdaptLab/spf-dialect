# helpers ==========================================================================================

# This does some fairly insane stuff. It creates and object file using the llc
# compiler from llvm called
# 	${pipeline_output_path without extension}_${pipeline}.o
# and then renames any symbol names in said object file from ${benchmark_name}
# to ${benchmark_name}_${pipeline}
macro(create_object_file_from_pipeline_output_and_append object_file_list pipeline_output_path benchmark_name pipeline)
    # Turn the filename into list [filename_no_extension, extension]
    # https://stackoverflow.com/a/5272993/3217397
    string(REPLACE "." ";" pipeline_out_path_parts ${pipeline_output_path})

    # Grab the fist thing from the list one
    # https://stackoverflow.com/a/56586927/3217397
    list(GET pipeline_out_path_parts 0 pipeline_output_path_no_extension)

    # translate from mlir llvm dialect to llvm bytecode
    set(bytecode_filename ${pipeline_output_path_no_extension}.bc)
    add_custom_command(OUTPUT ${bytecode_filename}
            DEPENDS ${pipeline_output_path}
            COMMAND mlir-translate
            -mlir-to-llvmir
            -o ${bytecode_filename}
            ${pipeline_output_path}
            WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

    # compile llvm bytecode to object file
    set(object_file_filename ${pipeline_output_path_no_extension}.o)
    add_custom_command(OUTPUT ${object_file_filename}
            DEPENDS ${bytecode_filename}
            COMMAND llc
            -relocation-model=pic
            -filetype=obj
            -O3
            -o ${object_file_filename}
            ${bytecode_filename}
            # objcopy redefines symbol in created object file. We're assuming
            # that someone is using the ciface machinery: https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
            COMMAND objcopy ${object_file_filename} --redefine-sym _mlir_ciface_${benchmark_name}=_mlir_ciface_${benchmark_name}_${pipeline}
            # We'll also need to rename the original function
            COMMAND objcopy ${object_file_filename} --redefine-sym ${benchmark_name}=${benchmark_name}_${pipeline}
            WORKING_DIRECTORY ${LLVM_BUILD_BINARY_DIR}/bin)

    # append to global list for later use in creating executable
    list(APPEND ${object_file_list} ${object_file_filename})
endmacro()

# Creates an intermediate mlir file with just the result of running the spf
# lowering, it's useful for debugging.
macro(create_partial_lowering out_partial_lowering_filename file_name_no_extension mlir_file)
    # lower benchmark out of mlir spf dialect
    set(partial_lowering_filename ${PROJECT_BINARY_DIR}/bench/${file_name_no_extension}_partial_lowering.mlir)
    set(${out_partial_lowering_filename} ${partial_lowering_filename})
    add_custom_command(OUTPUT ${partial_lowering_filename}
            DEPENDS ${mlir_file}
            COMMAND standalone-opt ${mlir_file}
            -my-pass
            -o ${partial_lowering_filename}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)


endmacro()

macro(get_file_name_no_extension out_param mlir_file)
    # Turn the filename into list [filename_no_extension, extension]
    # https://stackoverflow.com/a/5272993/3217397
    string(REPLACE "." ";" file_parts ${mlir_file})

    # Grab the fist thing from the list one
    # https://stackoverflow.com/a/56586927/3217397
    list(GET file_parts 0 file_name_no_extension)
    set(${out_param} ${file_name_no_extension})
endmacro()

# This sets up commands to create object file from the input MLIR file via the
# CPU pipeline and appends to MLIR_BENCHMARK_OBJECT_FILES list. It expects that
# there is an exported symbol called _mlir_ciface_${benchmark_name} in each mlir
# file. You'll end up with an object file called:
# 	${name without_extension from mlir_file}_${pipeline}_output.o
# the symbol _mlir_ciface_${benchmark_name} will be renamed to:
# _mlir_ciface_${benchmarkName}_${pipeline}
macro(append_cpu_pipeline_object_file object_file_list mlir_file benchmark_name)
    get_file_name_no_extension(file_name_no_extension ${mlir_file})

    create_partial_lowering(partial_lowering_filename ${file_name_no_extension} ${mlir_file})

    set(cpu_pipeline_filename ${PROJECT_BINARY_DIR}/bench/${file_name_no_extension}_cpu_pipeline_output.mlir)
    add_custom_command(OUTPUT ${cpu_pipeline_filename}
            DEPENDS ${partial_lowering_filename}
            COMMAND standalone-opt ${partial_lowering_filename}
            -inline
            -cse
            -loop-invariant-code-motion
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
            -o ${cpu_pipeline_filename}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

    create_object_file_from_pipeline_output_and_append(${object_file_list} ${cpu_pipeline_filename} ${benchmark_name} cpu)
endmacro()

# same thing as append_cpu_pipeline_object_file but for the GPU pipeline
macro(append_gpu_pipeline_object_file object_file_list mlir_file benchmark_name)
    get_file_name_no_extension(file_name_no_extension ${mlir_file})

    create_partial_lowering(partial_lowering_filename ${file_name_no_extension} ${mlir_file})

    set(gpu_pipeline_filename ${PROJECT_BINARY_DIR}/bench/${file_name_no_extension}_gpu_pipeline_output.mlir)
    add_custom_command(OUTPUT ${gpu_pipeline_filename}
                    DEPENDS ${partial_lowering_filename}
                    COMMAND standalone-opt ${partial_lowering_filename}
                    -inline
                    -cse
                    -lower-affine
                    -gpu-map-parallel-loops
                    -convert-parallel-loops-to-gpu
                    -lower-affine
                    -convert-vector-to-scf
                    -convert-scf-to-cf
                    -func-bufferize
                    -arith-bufferize
                    -finalizing-bufferize
                    -gpu-kernel-outlining
                    | ../build/bin/standalone-opt -pass-pipeline=\'builtin.module\(gpu.module\(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin\)\)\'
                    | ../build/bin/standalone-opt -gpu-async-region
                    -gpu-to-llvm
                    -convert-vector-to-llvm
                    -convert-memref-to-llvm
                    -convert-complex-to-standard
                    -convert-math-to-llvm
                    -convert-complex-to-llvm
                    -convert-math-to-libm
                    -convert-func-to-llvm
                    -reconcile-unrealized-casts
                    -o ${gpu_pipeline_filename}
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bench)

    create_object_file_from_pipeline_output_and_append(${object_file_list} ${gpu_pipeline_filename} ${benchmark_name} gpu)
endmacro()

# create benchmarks ================================================================================

# create object files for MLIR benchmarks
set(mlir_benchmark_object_files)
append_cpu_pipeline_object_file(mlir_benchmark_object_files sparse_mttkrp_cpu.mlir sparse_mttkrp)
append_gpu_pipeline_object_file(mlir_benchmark_object_files sparse_mttkrp_gpu.mlir sparse_mttkrp)
append_cpu_pipeline_object_file(mlir_benchmark_object_files sparse_ttm.mlir sparse_ttm)

# create bench executable
add_executable(bench ${mlir_benchmark_object_files} benchmarks.cpp benchmarks.h driver.cpp)
target_compile_options(bench PUBLIC "-O3")
target_link_libraries(bench
        mlir_c_runner_utils
        mlir_runner_utils
        mlir_cuda_runtime
        GPURuntime
        CPURuntime
        LLVMSupport
        )