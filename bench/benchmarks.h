
#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <cstdint>
#include <vector>

std::vector<int64_t> cpu_mttkrp_mlir(bool debug, int64_t iterations,
                                     char *filename);

std::vector<int64_t> cpu_mttkrp_iegenlib(bool debug, int64_t iterations,
                                         char *filename);

std::vector<int64_t> cpu_ttm_mlir(bool debug, int64_t iterations,
                                  char *filename);

std::vector<int64_t> cpu_ttm_iegenlib(bool debug, int64_t iterations,
                                      char *filename);

std::vector<int64_t> gpu_mttkrp_mlir(bool debug, int64_t iterations,
                                     char *filename);

std::vector<int64_t> gpu_ttm_mlir(bool debug, int64_t iterations,
                                  char *filename);

#endif // BENCHMARKS_H