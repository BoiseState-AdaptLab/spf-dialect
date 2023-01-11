#ifndef CPU_RUNTIME_H
#define CPU_RUNTIME_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <cstdint>
#include <vector>

struct COO {
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

public:
  const uint64_t nnz;
  const uint64_t rank;
  std::vector<uint64_t> dims;
  std::vector<std::vector<uint64_t>> coord;
  std::vector<double> values;
};

extern "C" {
char *getTensorFilename(uint64_t id);
void *_mlir_ciface_read_coo(char *filename);
void _mlir_ciface_coords(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                         uint64_t dim);
void _mlir_ciface_values(StridedMemRefType<double, 1> *ref, void *coo);
int64_t _mlir_ciface_nanoTime();
}

#endif // CPU_RUNTIME_H