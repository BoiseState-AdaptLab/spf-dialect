# An MLIR dialect for the Sparse Polyhedral Framework

By using [MLIR](https://mlir.llvm.org/), `spf-dialect` provides a more portable Sparse Polyhedral Framework (SPF) interface than the [existing SPF tools](http://dx.doi.org/10.1109/COMPSAC51774.2021.00275) it builds on. SPF is used in a wide variety of research settings. To name a few: SPF is used in synthesizing sparse tensor [format conversions](https://dl.acm.org/doi/10.1145/3579990.3580021) and
[contractions](https://dl.acm.org/doi/10.1145/3566054), and in [inspector-executor
compiler optimizations](https://ieeexplore.ieee.org/document/8436444). With this dialect researchers can express SPF in MLIR, and create SPF based optimizations inside MLIR based compilers. Because of the portability of MLIR code generation, SPF based tools can now target a much broader set of hardware than previously possible.

CPU and GPU code generated using `spf-dialect` is competitive with code generated using the SPF `C` front-end ([`IEGenLib`](https://github.com/BoiseState-AdaptLab/IEGenLib)), and code from the [PASTA benchmark suite](https://gitlab.com/tensorworld/pasta). The SPF `C` front-end is not capable of generating GPU code.
![performance comparison](https://user-images.githubusercontent.com/2278731/223522866-0bfbf2ae-7079-4b53-9552-2f4c24b7a2ea.png)


## Building
- Tested against `LLVM` 16 at sha: `570117b`.

This project builds with [`CMake`](https://cmake.org/) and depends on
[`LLVM`](https://github.com/llvm/llvm-project) (MLIR is part of LLVM), and
[`IEGenLib`](https://github.com/BoiseState-AdaptLab/IEGenLib). `IEGenLib` will be
automatically cloned and built during the `spf-dialect` build, however an existing source build of `LLVM` is
required before building `spf-dialect`.

The `LLVM` docs have great [instructions on how to build
LLVM](https://llvm.org/docs/CMake.html), though `spf-dialect` requires some
non-standard flags. `IEGenLib` requires `LLVM` to be built with exception handling and RTTI
(Run Time Type Information), which are [not part of a standard `LLVM`
build](https://llvm.org/docs/CodingStandards.html#do-not-use-rtti-or-exceptions).
GPU code generation also requires a few flags to be set. The `spf-dialect` tests
require a few `LLVM` utilities that aren't part of a regular build. An
example `LLVM` build with the required flags is shown here
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

Building `spf-dalect` (once `LLVM` is built) requires flags to set the path to the `LLVM` build, the path to the `lit` tool (built during `LLVM` build), and the same RTTI and execption handling flags as the `LLVM` build. This setup assumes that you have built `LLVM` (with MLIR enabled) in `$BUILD_DIR`. To build and launch the tests, run
```sh
mkdir build && cd build
# IEGenLib requires building with Makefiles and -j 1 for *reasons*
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
   -DLLVM_REQUIRES_EH=ON \
   -DLLVM_ENABLE_RTTI=ON
make
make check-spf
```
