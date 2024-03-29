#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
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
T defaultIfAbsent(const std::map<std::string, T> &map, char *key, T ifAbsent) {
  T out;
  if (map.find(key) != map.end()) {
    out = map.at(key);
  } else {
    out = ifAbsent;
  }

  return out;
}

// these functions will be named platform_benchmark_implementation e.g.
// cpu_tiledMTTKRP_mlir
typedef std::function<std::optional<std::vector<int64_t>>(Config config,
                                                          char *filename)>
    BenchmarkFunction;

// what platform to run benchmark on e.g. cpu, gpu
enum Platform : int { cpu = 0, gpu = 1, platform_not_found };
static const std::map<std::string, Platform> stringToPlatform{{"cpu", cpu},
                                                              {"gpu", gpu}};

// what benchmark to run (possibly with an optimization) e.g. mttkrp,
// tiled_mttkrp
enum Benchmark : int { mttkrp = 0, ttm = 1, benchmark_not_found = -1 };
static const std::map<std::string, Benchmark> stringToBenchmark{
    {"mttkrp", mttkrp}, {"ttm", ttm}};

//  what was used to implement the benchmark e.g. MLIR or IEGenLib
enum Implementation : int {
  mlir = 0,
  iegenlib = 1,
  implementation_not_found = -1
};
static const std::map<std::string, Implementation> stringToImplementation{
    {"mlir", mlir}, {"iegenlib", iegenlib}};

void printUsage(char *programName) {
  std::cerr << color[red] << "Expected 4 arguments\n"
            << color[green] << "Usage: " << programName
            << "<platform: cpu gpu> <benchmark: mttkrp ttm> "
               "<implementation: mlir, iegenlib> "
               "<filename (should be in matrix market exchange, or "
               "FROSST with extended header format)>\n"
            << color[reset];
}

int main(int argc, char *argv[]) {
  char *argProgramName = argv[0];

  if (argc != 5) {
    printUsage(argProgramName);
    exit(1);
  }

  char *argPlatform = argv[1];
  char *argBenchmark = argv[2];
  char *argImplementation = argv[3];
  char *argFilename = argv[4];

  // read benchmark out of command line arguments
  Benchmark benchmark;
  if ((benchmark = defaultIfAbsent(stringToBenchmark, argBenchmark,
                                   benchmark_not_found)) ==
      benchmark_not_found) {
    std::cerr << color[red] << "\"" << argBenchmark
              << "\" benchmark not found\n"
              << color[reset];

    printUsage(argProgramName);
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
    printUsage(argProgramName);
    exit(1);
  }

  // read platform out of command line arguments
  Platform platform;
  if ((platform = defaultIfAbsent(stringToPlatform, argPlatform,
                                  platform_not_found)) == platform_not_found) {
    std::cerr << color[red] << "\"" << argPlatform << "\" platform not found\n"
              << color[reset];
    printUsage(argProgramName);
    exit(1);
  }

  auto not_implemented = [](Config _,
                            char *__) -> std::optional<std::vector<int64_t>> {
    return std::nullopt;
  };

  std::vector<std::vector<std::vector<BenchmarkFunction>>> benchmarks{
      // CPU,
      {
          // MLIR,          IEGENLIB
          {cpu_mttkrp_mlir, cpu_mttkrp_iegenlib}, // MTTKRP
          {cpu_ttm_mlir, cpu_ttm_iegenlib},       // TTM
      },
      // GPU
      {
          // MLIR,          IEGENLIB
          {gpu_mttkrp_mlir, not_implemented}, // MTTKRP
          {gpu_ttm_mlir, not_implemented},    // MTTKRP
      },
  };

  // call the benchmark
  auto times =
      benchmarks[platform][benchmark][implementation](Config(), argFilename);

  // dump the results in csv
  if (times) {
    for (auto time : *times) {
      std::cout << argPlatform << ", " << argBenchmark << ", "
                << argImplementation << ", "
                << std::filesystem::path(argFilename).filename().string()
                << ", " << time << "\n";
    }
  }

  return 0;
}
