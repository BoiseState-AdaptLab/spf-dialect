# An MLIR dialect for the Sparse Polyhedral Framework

An [Multi-Level Intermediate Representation (MLIR)](https://mlir.llvm.org/)
dialect for the Sparse Polyhedral Framework (SPF). SPF has ben used in a wide
variety of research settings: in tools for synthesizing sparse tensor [format
conversions](https://dl.acm.org/doi/10.1145/3579990.3580021) and
[contractions](https://dl.acm.org/doi/10.1145/3566054), in [inspector-executor
compiler optimizations](https://ieeexplore.ieee.org/document/8436444), and many
more. This dialect builds on [existing SPF
tools](http://dx.doi.org/10.1109/COMPSAC51774.2021.00275) to provide an MLIR
interface, and MLIR code generation for SPF. Generating MLIR allows SPF based
tools to target a much broader set of hardware than previously possible, and
allows researchers to create SPF based optimizations inside MLIR based
compilers.

## Building
- Tested against LLVM 16 at sha: `570117b`.

This project builds with [CMake](https://cmake.org/) and depends on
[MLIR/LLVM](https://github.com/llvm/llvm-project), and
[IEGenLib](https://github.com/BoiseState-AdaptLab/IEGenLib). IEGenLib will be
automatically cloned and built during the build. A source build of LLVM is
required.

The LLVM docs have great [instructions on how to build
LLVM](https://llvm.org/docs/CMake.html), though `spf-dialect` requires some
non-standard flags. IEGenLib integration requires exception handling and RTTI
(Run Time Type Information) which are [not part of a standard
build](https://llvm.org/docs/CodingStandards.html#do-not-use-rtti-or-exceptions).
GPU code generation also requires a few flags to be set. The `spf-dialect` tests
require a few LLVM utilities that aren't part of a regular build. Here's an
example build of LLVM with the required flags:
```sh
# shallow clone of llvm to save time
git clone --depth 1 https://github.com/llvm/llvm-project.git

mkdir llvm-project/build && cd llvm-project/build

# To compile LLVM you first have to generate the build script with CMake. Taking
# a look at the flags:
# LLVM_ENABLE_PROJECTS="mlir"        - MLIR project must be enabled.
# LLVM_TARGETS_TO_BUILD="X86;NVPTX"  - NVPTX adds Nvidia GPU support.
# LLVM_ENABLE_EH=ON                  - Build llvm with exception handling. This
#                                      is disabled by default, using
#                                      fno-exceptions flag.
# LLVM_ENABLE_RTTI=ON                - Build llvm with C++ RTTI.
# LLVM_INSTALL_UTILS=ON              - Install utilities, like lit, needed for
#                                      testing spf-dialect.
# MLIR_ENABLE_CUDA_RUNNER=ON         - Required for Nvidia GPU code generation
#                                      from MLIR.
cmake -G Ninja ../llvm \
	 -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
	 -DLLVM_REQUIRES_EH=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_CUDA_RUNNER=ON

# build all targets
ninja
```

Building `spf-dalect` requires a few flags to let the build know where LLVM/MLIR
is. This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and
installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
# IEGenLib requires building with Makefiles and -j 1 for *reasons*
cmake -G "Unix Makefiles" .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
make
make check-spf
```
