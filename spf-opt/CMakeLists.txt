get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        SPFTransforms
        MLIROptLib
        MLIRSPF
        iegenlib
        gmp
        isl
        codegen
        omega
        )
add_llvm_executable(spf-opt spf-opt.cpp)

llvm_update_compile_flags(spf-opt)
target_link_libraries(spf-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(spf-opt)
