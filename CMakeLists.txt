cmake_minimum_required(VERSION 3.13.4)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

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
        GIT_TAG "9ece19271e991023000a6693efd46727b6c353f6"
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/installed
        BUILD_COMMAND ${MAKE})
## Add depeendence on GTEST
#FetchContent_Declare(
#        googletest
#        GIT_REPOSITORY https://github.com/google/googletest.git
#        GIT_TAG        release-1.8.0
#)
#
#FetchContent_GetProperties(googletest)
#if(NOT googletest_POPULATED)
#  FetchContent_Populate(googletest)
#  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
#endif()


# INCLUDE EXTERNAL
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/installed/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/installed/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib/lib/installed/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib/lib/installed/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/installed/include/iegenlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/IEGenLib/lib/isl/include)
#=======================================================================================================================


add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(standalone-opt)
add_subdirectory(standalone-translate)
