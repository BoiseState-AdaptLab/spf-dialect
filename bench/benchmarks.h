#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Config reads some environment variables for benchmarks. It should probably be
// in it's own header but whatever.
struct Config {
  // Number of columns of output matrix for MTTKRP
  uint64_t J = 5;
  // Number of columns of input matrix for TTM
  uint64_t R = 2;
  // The mode to hold constant for TTM
  uint64_t constantMode = 0;
  // debug mode prints all results
  bool debug = false;
  // test mode runs reference implementation and checks results
  bool test = false;
  // number of iterations to run the a given benchmark
  int64_t iterations = 5;

  Config() {
    if (std::getenv("MTTKRP_J")) {
      J = std::stoull(std::getenv("MTTKRP_J"));
    }
    if (std::getenv("TTM_R")) {
      R = std::stoull(std::getenv("TTM_R"));
    }
    if (getenv("ITERATIONS")) {
      iterations = std::stol(getenv("ITERATIONS"));
    }
    if (getenv("DEBUG")) {
      iterations = 1;
      debug = true;
    }
    if (getenv("TEST")) {
      iterations = 1;
      test = true;
    }
  }
};

std::vector<int64_t> cpu_mttkrp_mlir(Config config, char *filename);

std::vector<int64_t> cpu_mttkrp_iegenlib(Config config, char *filename);

std::vector<int64_t> cpu_ttm_mlir(Config config, char *filename);

std::vector<int64_t> cpu_ttm_iegenlib(Config config, char *filename);

std::vector<int64_t> gpu_mttkrp_mlir(Config config, char *filename);

std::vector<int64_t> gpu_ttm_mlir(Config config, char *filename);

#endif // BENCHMARKS_H