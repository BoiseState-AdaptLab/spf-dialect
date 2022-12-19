// Parts of this code are taken from
// https://github.com/llvm/llvm-project/blob/b682616d1fd1263b303985b9f930c1760033af2c/mlir/lib/ExecutionEngine/SparseTensorUtils.cpp
// Which is part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index". For now, we simply assume that
/// type is 64-bit, but targets with different "index" bit widths should
/// link with an alternatively built runtime support library.
// TODO: support such targets?
using index_type = uint64_t;

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

struct COO {
  COO(uint64_t nnz, uint64_t rank) {
    coord =
        std::vector<std::vector<uint64_t>>(rank, std::vector<uint64_t>(nnz));
    values = std::vector<double>(nnz);
  }

public:
  std::vector<std::vector<uint64_t>> coord;
  std::vector<double> values;
};

extern "C" { // these are the symbols that MLIR will actually call

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
  for (index_type i = 0; i < rank; i++) {
    dims[i] = idata[i + 2];
  }

  COO *coo = new COO(nnz, rank);

  std::vector<std::vector<index_type>> &coord = coo->coord;
  std::vector<double> &values = coo->values;

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

void _mlir_ciface_coords(StridedMemRefType<index_type, 1> *ref, void *coo,
                         index_type dim) {
  std::vector<index_type> &v = static_cast<COO *>(coo)->coord[dim];
  ref->basePtr = ref->data = v.data();
  ref->offset = 0;
  ref->sizes[0] = v.size();
  ref->strides[0] = 1;
}

void _mlir_ciface_values(StridedMemRefType<double, 1> *ref, void *coo) {
  std::vector<double> &v = static_cast<COO *>(coo)->values;
  ref->basePtr = ref->data = v.data();
  ref->offset = 0;
  ref->sizes[0] = v.size();
  ref->strides[0] = 1;
}

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
char *getTensorFilename(index_type id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  if (!env)
    FATAL("Environment variable %s is not set\n", var);
  printf("filename: %s\n", env);
  return env;
}

void _mlir_ciface_print2D(UnrankedMemRefType<index_type> *store) {
  auto m = DynamicMemRefType<index_type>(*store);
  printf("m.rank: %lu\n", m.rank);

  printf("m.sizes: [");
  for (int i = 0; i < m.rank; i++) {
    if (i != 0) {
      printf(",");
    }
    printf("%lu", m.sizes[i]);
  }
  printf("]\n");

  printf("data(linear): [");
  for (int i = 0; i < m.sizes[0] * m.sizes[1]; i++) {
    if (i != 0) {
      printf(",");
    }
    printf("%ld", m.data[i]);
  }
  printf("]\n");

  printf("data: [");
  for (int i = 0; i < m.sizes[0]; i++) {
    if (i != 0) {
      printf(",\n       ");
    }
    printf("[");

    for (int j = 0; j < m.sizes[1]; j++) {
      if (j != 0) {
        printf(",");
      }
      printf("%ld", m.data[i * m.sizes[1] + j]);
    }
    printf("]");
  }
  printf("]\n");
}

void _mlir_ciface_printStatementCalls(UnrankedMemRefType<index_type> *store) {
  auto m = DynamicMemRefType<index_type>(*store);
  for (int i = 0; i < m.sizes[0]; i++) {
    // if (i != 0) {
    //   printf(",\n       ");
    // }
    // printf("[");

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