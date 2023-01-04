#ifndef RUNTIME_H
#define RUNTIME_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <cstdint>
#include <vector>


/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index". For now, we simply assume that
/// type is 64-bit, but targets with different "index" bit widths should
/// link with an alternatively built runtime support library.
// TODO: support such targets?
using index_type = uint64_t;

struct COO {
  COO(const uint64_t nnz, const uint64_t rank, std::vector<uint64_t> &&dims)
      : nnz(nnz), rank(rank), dims(std::move(dims)) {
    assert(this->dims.size() == rank && "dims.size() != rank in COO constructor");
    coord =
        std::vector<std::vector<uint64_t>>(rank, std::vector<uint64_t>(nnz));
    values = std::vector<double>(nnz);
  }

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

#endif // RUNTIME_H
