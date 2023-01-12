
#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <cstdint>

int64_t cpu_mttkrp_iegenlib(bool debug, int64_t iterations, char *filename);

int64_t cpu_mttkrp_mlir(bool debug, int64_t iterations, char *filename);

int64_t gpu_mttkrp_mlir(bool debug, int64_t iterations, char *filename);

#endif // BENCHMARKS_H