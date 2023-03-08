# bench

Contains the benchmarking harness. C code generated from [C front
end](https://github.com/BoiseState-AdaptLab/IEGenLib) is included directly in
[`benchmarks.cpp`](https://github.com/BoiseState-AdaptLab/spf-dialect/blob/main/bench/benchmarks.cpp)
(which contains implementations of each benchmark called by main driver
[`driver.cpp`](https://github.com/BoiseState-AdaptLab/spf-dialect/blob/main/bench/driver.cpp)).
MLIR kernels are compiled to object files with
[`spf-opt`](https://github.com/BoiseState-AdaptLab/spf-dialect/tree/main/spf-opt),
[`mlir-translate`](https://github.com/llvm/llvm-project/tree/main/mlir/tools/mlir-translate),
and [`llc`](https://llvm.org/docs/CommandGuide/llc.html). [The
build](https://github.com/BoiseState-AdaptLab/spf-dialect/blob/main/bench/CMakeLists.txt),
which also sets up the compilation pipeline for MLIR kernels, links the
resulting object files with `bench`.