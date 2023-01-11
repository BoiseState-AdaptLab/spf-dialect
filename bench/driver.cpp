#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "benchmarks.h"

// Stuff for printing to the console.
enum {
  reset = 0,
  red,
  green,
  yellow,
  blue,
  purple,
  cyan,
  gray,
  white,
};
const char *color[] = {"\033[0m",  "\033[31m", "\033[32m",
                       "\033[33m", "\033[34m", "\033[35m",
                       "\033[36m", "\033[37m", "\033[97m"};

// Returns value if <key> is present in map otherwise returns <ifAbsent>.
template <typename T>
T defaultIfAbsent(std::map<std::string, T> &map, char *key, T ifAbsent) {
  T out;
  if (map.find(key) != map.end()) {
    out = map[key];
  } else {
    out = ifAbsent;
  }

  return out;
}

// these functions will be named platform_benchmark_implementation e.g.
// cpu_tiledMTTKRP_mlir
typedef std::function<double(bool debug, char *filename)> BenchmarkFunction;

void printUsage(char *programName) {
  std::cerr << color[red] << "Expected 4 arguments\n"
            << color[green] << "Usage: " << programName
            << " <filename (should be in matrix market exchange, or "
               "FROSST with extended header format)> "
               "<platform: cpu nvidia-gpu> <benchmark: coo_mttkrp> "
               "<implementation: mlir, iegenlib, pasta>\n"
            << color[reset];
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printUsage(argv[0]);
    exit(1);
  }

  char *programName = argv[0];
  char *filename = argv[1];
  char *argPlatform = argv[2];
  char *argBenchmark = argv[3];
  char *argImplementation = argv[4];

  // debug flag is set via environment variable
  bool debug = false;
  if (getenv("DEBUG")) {
    debug = true;
  }

  // what platform to run benchmark on e.g. cpu, gpu
  enum Platform : int { cpu = 0, nvidia_gpu = 1, platform_not_found };
  std::map<std::string, Platform> stringToPlatform{{"cpu", cpu},
                                                   {"gpu", nvidia_gpu}};

  // what benchmark to run (possibly with an optimization) e.g. mttkrp,
  // tiled_mttkrp
  enum Benchmark : int { mttkrp = 0, benchmark_not_found = -1 };
  std::map<std::string, Benchmark> stringToBenchmark{{"mttkrp", mttkrp}};

  //  what was used to implement the benchmark e.g. MLIR, IEGenLib, or PASTA
  enum Implementation : int {
    mlir = 0,
    iegenlib = 1,
    implementation_not_found = -1
  };
  std::map<std::string, Implementation> stringToImplementation{
      {"mlir", mlir}, {"iegenlib", iegenlib}};

  // read benchmark out of command line arguments
  Benchmark benchmark;
  if ((benchmark = defaultIfAbsent(stringToBenchmark, argBenchmark,
                                   benchmark_not_found)) ==
      benchmark_not_found) {
    std::cerr << color[red] << "\"" << argBenchmark
              << "\" benchmark not found\n"
              << color[reset];
    printUsage(programName);
    exit(1);
  }

  // read implementation out of command line arguments
  Implementation implementation;
  if ((implementation = defaultIfAbsent(
           stringToImplementation, argImplementation,
           implementation_not_found)) == implementation_not_found) {
    std::cerr << color[red] << "\"" << argImplementation
              << "\" implementation not found\n"
              << color[reset];
    printUsage(programName);
    exit(1);
  }

  // read platform out of command line arguments
  Platform platform;
  if ((platform = defaultIfAbsent(stringToPlatform, argPlatform,
                                  platform_not_found)) == platform_not_found) {
    std::cerr << color[red] << "\"" << argPlatform << "\" platform not found\n"
              << color[reset];
    printUsage(programName);
    exit(1);
  }

  auto not_implemented = [](bool debug, char *_) -> double {
    if (debug) {
      std::cout << "not implemented\n";
    }
    return -1;
  };

  // benchmarks stored in Platform x Benchmark x Implementation vector
  std::vector<std::vector<std::vector<BenchmarkFunction>>> benchmarks{
      // CPU,
      {
          // MLIR,                  IEGENLIB
          {cpu_mttkrp_mlir, cpu_mttkrp_iegenlib}, // MTTKRP
      },
      // GPU
      {
          // MLIR,                  IEGENLIB
          {gpu_mttkrp_mlir, not_implemented}, // MTTKRP
      },
  };

  // Call the benchmark
  benchmarks[platform][benchmark][implementation](debug, filename);

  return 0;
}