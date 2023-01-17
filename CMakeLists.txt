cmake_minimum_required(VERSION 3.13.4)
project(standalone-dialect LANGUAGES CXX C CUDA)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#ADD EXTERNAL IEGEN LIB ================================================================================================
include(ExternalProject)
ExternalProject_Add(iegenlib_in
        GIT_REPOSITORY "https://github.com/BoiseState-AdaptLab/IEGenLib.git"
        GIT_TAG "ad14a4fe6f078e0cc3e0c3944843ff14f9785269"
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/installed
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ${MAKE})

# INCLUDE EXTERNAL
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib/lib/installed/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib/lib/installed/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/installed/include/iegenlib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/installed/lib)

# subdirectories =======================================================================================================
add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(standalone-opt)
add_subdirectory(parser)
add_subdirectory(standalone-translate)
add_subdirectory(jacobi)
add_subdirectory(bench)
