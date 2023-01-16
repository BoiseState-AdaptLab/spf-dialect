#ifndef CPU_RUNTIME_H
#define CPU_RUNTIME_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <cstdint>
#include <vector>

typedef std::vector<std::vector<uint64_t>> coord_t;

struct COO {
  const uint64_t nnz;
  const uint64_t rank;
  std::vector<uint64_t> dims;
  coord_t coord;
  std::vector<double> values;

  COO(const COO&) = delete;

  COO(const uint64_t nnz, const uint64_t rank, std::vector<uint64_t> &&dims)
      : nnz(nnz), rank(rank), dims(std::move(dims)) {
    assert(this->dims.size() == rank &&
           "dims.size() != rank in COO constructor");
    coord =
        std::vector<std::vector<uint64_t>>(rank, std::vector<uint64_t>(nnz));
    values = std::vector<double>(nnz);
  }

  COO(const uint64_t nnz, const uint64_t rank, std::vector<uint64_t> &&dims,
      std::vector<std::vector<uint64_t>> &&coord, std::vector<double> &&values)
      : nnz(nnz), rank(rank), dims(std::move(dims)), coord(std::move(coord)),
        values(std::move(values)) {}

  // Sort indices lexigraphically except consider <mode> as if it were the last
  // mode. This is useful for computing the fibers of a sparse matrix.
  void sortIndicesModeLast(uint64_t modeLast) {
    sortIndicesModeLast(modeLast, 0, nnz - 1);
  }

  void dump(std::ostream &out) {
    for (uint64_t i = 0; i < nnz; i++) {
      for (uint64_t dim = 0; dim < rank; dim++) {
        out << coord[dim][i] << " ";
      }
      out << ": " << values[i] << "\n";
    }
  }


private:
  // Sort indices lexigraphically considering <mode> the last one
  // TODO: create a random access iterator that can view the data each entry at
  // a time rather than along entries and use std::sort instead.
  void sortIndicesModeLast(uint64_t modeLast, uint64_t lower, uint64_t upper) {
    if (lower >= upper) {
      return;
    }
    uint64_t pivot = upper;
    int64_t i = lower;

    for (uint64_t j = lower; j < pivot; j++) {
      if (compareCoordsModeLast(modeLast, j, pivot)) {
        swap(i, j);
        i++;
      }
    }
    swap(i, pivot);

    sortIndicesModeLast(modeLast, lower, i != 0 ? i - 1 : 0);
    sortIndicesModeLast(modeLast, i + 1, upper);
  }

  // compareCoordsModeLast returns true if first is lexigraphically less than
  // second, treat modeLast as the last mode regardless of position.
  bool compareCoordsModeLast(uint64_t modeLast, uint64_t first,
                             uint64_t second) {
    // lexicographic sort based on coordinate values
    for (uint64_t m = 0; m < rank; m++) {
      if (m != modeLast) {
        auto one = coord[m][first];
        auto two = coord[m][second];
        if (one != two) {
          return one < two;
        }
      }
    }
    return coord[modeLast][first] < coord[modeLast][second];
  }

  void swap(uint64_t i, uint64_t j) {
    if (i == j) {
      return;
    }
    // TODO: maybe template this out to avoid heap allocation
    std::vector<uint64_t> tmpCoords(rank);
    for (uint64_t mode = 0; mode < rank; mode++) {
      tmpCoords[mode] = coord[mode][i];
    }

    for (uint64_t mode = 0; mode < rank; mode++) {
      coord[mode][i] = coord[mode][j];
    }

    for (uint64_t mode = 0; mode < rank; mode++) {
      coord[mode][j] = tmpCoords[mode];
    }

    double tmpVal = values[i];
    values[i] = values[j];
    values[j] = tmpVal;
  }
};

extern "C" {
int64_t milliTime();

void *_mlir_ciface_read_coo(char *filename);

void _mlir_ciface_coords(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                         uint64_t dim);

void _mlir_ciface_values(StridedMemRefType<double, 1> *ref, void *coo);
}

#endif // CPU_RUNTIME_H