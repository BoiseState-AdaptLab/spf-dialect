add_executable(mttkrp mttkrp.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(mttkrp
        ${dialect_libs}
        iegenlib
        gmp
        isl
        codegen
        omega
        MLIRIR
        )
include_directories(mttkrp "lib/installed/include/iegenlib")
include_directories(mttkrp "lib/IEGenLib/lib/installed/include")
include_directories(mttkrp "lib/IEGenLib/lib/gtest/include")

#include_directories(${MLIR_INCLUDE_DIRS})
#link_directories(${LLVM_BUILD_LIBRARY_DIR})
