#ifndef CPU_RUNTIME_H
#define CPU_RUNTIME_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <algorithm> // std::sort
#include <array>
#include <cstdint>
#include <string>
#include <vector>

typedef std::vector<std::vector<uint64_t>> coord_t;

struct COO {
  const uint64_t nnz;
  const uint64_t rank;
  std::vector<uint64_t> dims;
  coord_t coord;
  std::vector<float> values;

  COO(const COO &) = delete;

  COO(const uint64_t nnz, const uint64_t rank, std::vector<uint64_t> &&dims)
      : nnz(nnz), rank(rank), dims(std::move(dims)) {
    assert(this->dims.size() == rank &&
           "dims.size() != rank in COO constructor");
    coord =
        std::vector<std::vector<uint64_t>>(rank, std::vector<uint64_t>(nnz));
    values = std::vector<float>(nnz);

    // populate column views. This has to be done carefully to avoid moving the
    // objects, see the move constructors for details.
    rowViews.reserve(nnz);
    for (size_t i = 0; i < nnz; i++) {
      rowViews.emplace_back(std::array<uint64_t *, 3>{coord[0].data() + i,
                                                      coord[1].data() + i,
                                                      coord[2].data() + i},
                            values.data() + i);
    }
  }

  COO(const uint64_t nnz, const uint64_t rank, std::vector<uint64_t> &&dims,
      std::vector<std::vector<uint64_t>> &&coord, std::vector<float> &&values)
      : nnz(nnz), rank(rank), dims(std::move(dims)), coord(std::move(coord)),
        values(std::move(values)) {}

  // Sort indices lexigraphically except consider <mode> as if it were the last
  // mode. This is useful for computing the fibers of a sparse matrix.
  void sortIndicesModeLast(uint64_t modeLast) {
    std::sort(this->rowViews.begin(), this->rowViews.end(),
              [=](RowView<3> &first, RowView<3> &second) {
                // lexicographic sort based on coordinate values
                for (uint64_t mode = 0; mode < rank; mode++) {
                  if (mode != modeLast) {
                    auto one = *first.coordPointers[mode];
                    auto two = *second.coordPointers[mode];
                    if (one != two) {
                      return one < two;
                    }
                  }
                }
                return *first.coordPointers[modeLast] <
                       *second.coordPointers[modeLast];
              });
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
  // RowView allows a row wise view of the COO data that we store column
  // wise. This is useful for sorting the data row wise. We store the data by
  // column because most benchmarks want it that way. Sorting is just a
  // pre-processing step.
  //
  // The approach is a little wasteful of memory, we store essentially a N
  // element fat pointer as well as temporary storage that might not even be
  // needed.  It doesn't really matter though, the benchmark machine has
  // multiple 100s of gigs of memory. We could store all the data one big array
  // and only store 1 pointer with a stride and maybe even have a bump allocator
  // for the temporary storage, but it's not worth it currently.
  template <uint8_t N> struct RowView {
    std::array<uint64_t, N> coordTempStorage;
    std::array<uint64_t *, N> coordPointers;
    float valueTempStorage;
    float *valuePointer;

    RowView(std::array<uint64_t *, N> pointers, float *valuePointer)
        : coordPointers(pointers), valuePointer(valuePointer) {}
    RowView(const RowView &other) = delete;
    RowView &operator=(const RowView &other) = delete;

    // In this case the object we're constructing isn't a view into any storage
    // so we save off the values to temporary storage.
    RowView(RowView &&other) noexcept {
      for (size_t i = 0; i < N; i++) {
        coordTempStorage[i] = *other.coordPointers[i];
      }
      for (size_t i = 0; i < N; i++) {
        coordPointers[i] = &coordTempStorage[i];
      }
      valueTempStorage = *other.valuePointer;
      valuePointer = &valueTempStorage;
    }

    // Move assignment operator is called for an existing object that already
    // has backing storage. Since this is a view not a value type that owns it's
    // data we don't want to steal the backing storage from the passed in
    // lvalue, just copy it over.
    RowView &operator=(RowView &&other) noexcept {
      assert(coordPointers.size() == other.coordPointers.size());
      // std::cout << "move assignment operator\n";
      for (size_t i = 0; i < N; i++) {
        *this->coordPointers[i] = *other.coordPointers[i];
      }
      *valuePointer = *other.valuePointer;
      return *this;
    }
  };
  std::vector<RowView<3>> rowViews;
};

std::vector<uint64_t> fiberStartStopIndices(COO &sortedCoo,
                                            uint64_t constantMode);

extern "C" {
int64_t milliTime();

void *_mlir_ciface_read_coo(char *filename);

void _mlir_ciface_coords(StridedMemRefType<uint64_t, 1> *ref, void *coo,
                         uint64_t dim);

void _mlir_ciface_values(StridedMemRefType<float, 1> *ref, void *coo);
}

#endif // CPU_RUNTIME_H