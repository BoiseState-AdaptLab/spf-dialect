// Parts of this code are taken from
// https://github.com/llvm/llvm-project/blob/b682616d1fd1263b303985b9f930c1760033af2c/mlir/lib/ExecutionEngine/SparseTensorUtils.cpp
// Which is part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.

#include "Runtime/CPURuntime.h"
#include <cassert>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// This macro helps minimize repetition of this idiom, as well as ensuring
// we have some additional output indicating where the error is coming from.
// (Since `fprintf` doesn't provide a stacktrace, this helps make it easier
// to track down whether an error is coming from our code vs somewhere else
// in MLIR.)
#define FATAL(...)                                                             \
  {                                                                            \
    fprintf(stderr, "SparseTensorUtils: " __VA_ARGS__);                        \
    exit(1);                                                                   \
  }

/// Helper to convert string to lower case.
static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

static constexpr int kColWidth = 1025;

/// Read the MME header of a general sparse matrix of type real.
static void readMMEHeader(FILE *file, char *filename, char *line,
                          uint64_t *idata, bool *isPattern, bool *isSymmetric) {
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5)
    FATAL("Corrupt header in %s\n", filename);
  // Set properties
  *isPattern = (strcmp(toLower(field), "pattern") == 0);
  *isSymmetric = (strcmp(toLower(symmetry), "symmetric") == 0);
  // Make sure this is a general sparse matrix.
  if (strcmp(toLower(header), "%%matrixmarket") ||
      strcmp(toLower(object), "matrix") ||
      strcmp(toLower(format), "coordinate") ||
      (strcmp(toLower(field), "real") && !(*isPattern)) ||
      (strcmp(toLower(symmetry), "general") && !(*isSymmetric)))
    FATAL("Cannot find a general sparse matrix in %s\n", filename);
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file))
      FATAL("Cannot find data in %s\n", filename);
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3)
    FATAL("Cannot find size in %s\n", filename);
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
static void readExtFROSTTHeader(FILE *file, char *filename, char *line,
                                uint64_t *idata) {
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file))
      FATAL("Cannot find data in %s\n", filename);
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2)
    FATAL("Cannot find metadata in %s\n", filename);
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; r++)
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1)
      FATAL("Cannot find dimension size %s\n", filename);
  fgets(line, kColWidth, file); // end of line
}

// areCoordsEqualExceptMode returns true if coords are equal in all modes
// <exceptMode>
bool areCoordsEqualExceptMode(COO &coo, uint64_t exceptMode, uint64_t i,
                              uint64_t j) {
  for (uint64_t mode = 0; mode < coo.rank; mode++) {
    if (mode != exceptMode) {
      auto one = coo.coord[mode][i];
      auto two = coo.coord[mode][j];
      if (one != two) {
        return false;
      }
    }
  }
  return true;
}

// fiberStartStopIndices returns the indices at which fibers in COO tensor
// <sortedCoo> formed by holding <constantMode> constant begin and end.
// <sortedCoo> is assumed to have been sorted lexigraphically with
// <constantMode> considered that last mode.
//
// Ex: the following COO mode 3 tensor is sorted lexigraphically with mode 0
//    (constant mode) last
// mode: 0,1,2
// 0|    1 0 0 : 77
// 1|    0 0 2 : 3
// 2|    1 0 2 : 10
// 3|    0 0 3 : 63
// ⬑ index
// output will be: [0,1,3,4].
//  - Fiber 0 starts at index 0 (always)
//  - Fiber 0 ends (and fiber 1 starts) at index 1 (as at index 1, one two one
//    of the non-constant dimensions from index 0 differ at mode 2: 0 to 2)
//  - Fiber 1 (and fiber 2 starts) ends at index 3 (as at index 3 one of the two
//    non-constant dimensions that are the same for index 1 and 2 differ at mode
//    2: 2 to 3)
//  - As fiber 2 is the last fiber it ends at the last index +1;
std::vector<uint64_t> fiberStartStopIndices(COO &sortedCoo,
                                            uint64_t constantMode) {
  std::vector<uint64_t> out;
  uint64_t lastIdx = sortedCoo.nnz;
  for (uint64_t i = 0; i < sortedCoo.nnz; i++) {
    if (lastIdx == sortedCoo.nnz ||
        !areCoordsEqualExceptMode(sortedCoo, constantMode, lastIdx, i)) {
      lastIdx = i;
      out.push_back(i);
    }
  }
  out.push_back(sortedCoo.nnz);
  return out;
}

extern "C" { // these are the symbols that MLIR will actually call

int64_t milliTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  return milliseconds.count();
}

void *_mlir_ciface_read_coo(char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Cannot find %s\n", filename);
    exit(1);
  }
  char line[kColWidth];
  uint64_t idata[512];
  bool isSymmetric = false;
  bool isPattern = false;
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, line, idata, &isPattern, &isSymmetric);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, line, idata);
  } else {
    fprintf(stderr, "Unknown format %s\n", filename);
    exit(1);
  }
  uint64_t rank = idata[0];
  uint64_t nnz = idata[1];

  auto dims = std::vector<uint64_t>(rank);
  for (uint64_t i = 0; i < rank; i++) {
    dims[i] = idata[i + 2];
  }

  COO *coo = new COO(nnz, rank, std::move(dims));

  std::vector<std::vector<uint64_t>> &coord = coo->coord;
  std::vector<float> &values = coo->values;

  // Read file into vectors
  for (uint64_t k = 0; k < nnz; k++) {
    if (!fgets(line, kColWidth, file)) {
      fprintf(stderr, "Cannot find next line of data in %s\n", filename);
      exit(1);
    }
    char *linePtr = line;
    for (uint64_t r = 0; r < rank; r++) {
      uint64_t idx = strtoul(linePtr, &linePtr, 10);
      coord[r][k] = idx - 1;
    }

    double value = strtod(linePtr, &linePtr);
    values[k] = value;
  }

  return coo;
}

void _mlir_ciface_coords(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                         uint64_t dim) {
  std::vector<uint64_t> &v = static_cast<COO *>(coo)->coord[dim];
  ref->basePtr = ref->data = v.data();
  ref->offset = 0;
  ref->sizes[0] = v.size();
  ref->strides[0] = 1;
}

void _mlir_ciface_values(StridedMemRefType<float, 1> *ref, void *coo) {
  std::vector<float> &v = static_cast<COO *>(coo)->values;
  ref->basePtr = ref->data = v.data();
  ref->offset = 0;
  ref->sizes[0] = v.size();
  ref->strides[0] = 1;
}

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
char *getTensorFilename(uint64_t id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  if (!env)
    FATAL("Environment variable %s is not set\n", var);
  return env;
}

void _mlir_ciface_printStatementCalls(UnrankedMemRefType<uint64_t> *store) {
  auto m = DynamicMemRefType<uint64_t>(*store);
  for (int i = 0; i < m.sizes[0]; i++) {
    printf("s%ld(", m.data[i * m.sizes[1]]);
    for (int j = 1; j < m.sizes[1]; j++) {
      if (j != 1) {
        printf(",");
      }
      printf("%ld", m.data[i * m.sizes[1] + j]);
    }
    printf(")\n");
  }
}
} // extern "C"