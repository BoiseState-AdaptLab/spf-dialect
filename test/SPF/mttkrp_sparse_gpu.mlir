// RUN: spf-opt %s \
// RUN:  -convert-spf-to-loops \
// RUN:  -inline \
// RUN:  -cse \
// RUN:  -lower-affine \
// RUN:  -gpu-map-parallel-loops \
// RUN:  -convert-parallel-loops-to-gpu \
// RUN:  -lower-affine \
// RUN:  -convert-vector-to-scf \
// RUN:  -convert-scf-to-cf \
// RUN:  -func-bufferize \
// RUN:  -arith-bufferize \
// RUN:  -finalizing-bufferize \
// RUN:  -gpu-kernel-outlining \
// RUN:  | spf-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' \
// RUN:  | spf-opt -gpu-async-region \
// RUN:  -gpu-to-llvm \
// RUN:  -convert-vector-to-llvm \
// RUN:  -convert-memref-to-llvm \
// RUN:  -convert-complex-to-standard \
// RUN:  -convert-math-to-llvm \
// RUN:  -convert-complex-to-llvm \
// RUN:  -convert-math-to-libm \
// RUN:  -convert-func-to-llvm \
// RUN:  -reconcile-unrealized-casts \
// RUN:  | TENSOR0="%spf_src_dir/test/data/mttkrp_b.tns" mlir-cpu-runner \
// RUN:    -entry-point-result=void \
// RUN:    -shared-libs=%spf_lib_dir/Runtime/libCPURuntime%shlibext \
// RUN:    -shared-libs=%spf_lib_dir/Runtime/libGPURuntime%shlibext \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:    | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
func.func private @coords_gpu(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
func.func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
func.func private @values_gpu(!llvm.ptr<i8>) -> memref<?xf32> attributes {llvm.emit_c_interface}

func.func private @UFi(%uf_argb_coord_0 : memref<?xindex>,
                        %uf_argb_coord_1 : memref<?xindex>,
                        %uf_argb_coord_2 : memref<?xindex>,
                        %z: index)-> index {
        %i = memref.load %uf_argb_coord_0[%z] : memref<?xindex>
        return %i : index
}

func.func private @UFk(%uf_argb_coord_0 : memref<?xindex>,
                        %uf_argb_coord_1 : memref<?xindex>,
                        %uf_argb_coord_2 : memref<?xindex>,
                        %z : index) -> index {
        %k = memref.load %uf_argb_coord_1[%z] : memref<?xindex>
        return %k : index
}

func.func private @UFl(%uf_argb_coord_0 : memref<?xindex>,
                        %uf_argb_coord_1 : memref<?xindex>,
                        %uf_argb_coord_2 : memref<?xindex>,
                        %z : index) -> index {
        %l = memref.load %uf_argb_coord_2[%z] : memref<?xindex>
        return %l : index
}

func.func @sparse_mttkrp(%NNZ : index,
                            %J: index,
                            %argb_coord_0 : memref<?xindex>,
                            %argb_coord_1 : memref<?xindex>,
                            %argb_coord_2 : memref<?xindex>,
                            %argb_values : memref<?xf32>,
                            %argc: memref<?x?xf32>,
                            %argd: memref<?x?xf32>,
                            %arga: memref<?x?xf32>) -> () {
    "spf.computation"() ({
        // COO MTTKRP:
        //
        // parallel_for (int j = 0; j < J; j++)
        //   for(int z = 0; z < NNZ; z++) {
        //     i=UFi(z);
        //     k=UFk(z);
        //     l=UFl(z);
        //     val=UFval(z);
        //     A[i,j] += val*D[l,j]*C[k,j];
        // }
        "spf.statement"(%NNZ, %J, %argb_coord_0, %argb_coord_1, %argb_coord_2, %argb_values, %argc, %argd, %arga) ({
        ^bb0(%b_i_k_l : f32, %c_k_j : f32, %d_l_j : f32, %a_i_j : f32):
        %0 = arith.mulf %b_i_k_l, %d_l_j : f32
        %1 = arith.mulf %0, %c_k_j : f32
        %2 = arith.addf %1, %a_i_j : f32
        "spf.yield"(%2) : (f32) -> ()
        })  {
                reads = [
                    [affine_map<(j, z, i, k, l) -> (z)>],
                    [affine_map<(j, z, i, k, l) -> (k, j)>],
                    [affine_map<(j, z, i, k, l) -> (l, j)>]
                ],
                writes = [
                    [affine_map<(j, z, i, k, l) -> (i, j)>]
                ],
                // symbols,ufInputs,inputs,outputs
                operand_segment_sizes = array<i32: 2,3,3,1>,
                symbolNames = ["NNZ", "J"],
                iteratorTypes = ["parallel", "reduction", "reduction", "reduction", "reduction"],
                executionSchedule = "{[j,z,i,k,l]->[j,z,i,k,l]}",
                iterationSpace = "{[j,z,i,k,l]: 0<=j<J and 0<=z<NNZ and i=UFi(z) and k=UFk(z) and l=UFl(z)}",
                transforms = []
            } : (index, index,
                    memref<?xindex>, memref<?xindex>,
                    memref<?xindex>, memref<?xf32>,
                    memref<?x?xf32>, memref<?x?xf32>,
                    memref<?x?xf32>) -> ()
    }) : () -> ()

    return
}

func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // dimensions of matrices for mttkrp_b
    %I = arith.constant 2 : index
    %J = arith.constant 5 : index
    %K = arith.constant 3 : index
    %L = arith.constant 4 : index
    %nnz = arith.constant 17 : index

    // // dimensions of matrices for nell-2-modified.tns
    // %I = arith.constant 12092 : index
    // %J = arith.constant 500 : index
    // %K = arith.constant 9184 : index
    // %L = arith.constant 28818 : index
    // %nnz = arith.constant 5879419 : index

    // Read the sparse B input from a file.
    %filename = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %storage = call @read_coo(%filename) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>

    %b_coord_0 = call @coords_gpu(%storage, %c0) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)

    %b_coord_1 = call @coords_gpu(%storage, %c1) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)

    %b_coord_2 = call @coords_gpu(%storage, %c2) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)

    %b_values = call @values_gpu(%storage) : (!llvm.ptr<i8>) -> (memref<?xf32>)

    // Initialize dense C and D inputs and dense output A.
    %c = memref.alloc(%K, %J) : memref<?x?xf32>
    scf.for %k = %c0 to %K step %c1 {
        scf.for %j = %c0 to %J step %c1 {
            %v0 = arith.muli %k, %J : index
            %v1 = arith.addi %v0, %j : index
            %v2 = arith.index_cast %v1 : index to i32
            %v = arith.sitofp %v2 : i32 to f32
            memref.store %v, %c[%k, %j] : memref<?x?xf32>
        }
    }
    %d_c = gpu.alloc(%K, %J) : memref<?x?xf32>
    gpu.memcpy %d_c, %c : memref<?x?xf32>, memref<?x?xf32>

    %d = memref.alloc(%L, %J) : memref<?x?xf32>
    scf.for %l = %c0 to %L step %c1 {
        scf.for %j = %c0 to %J step %c1 {
            %v0 = arith.muli %l, %J : index
            %v1 = arith.addi %v0, %j : index
            %v2 = arith.index_cast %v1 : index to i32
            %v = arith.sitofp %v2 : i32 to f32
            memref.store %v, %d[%l, %j] : memref<?x?xf32>
        }
    }
    %d_d = gpu.alloc(%L, %J) : memref<?x?xf32>
    gpu.memcpy %d_d, %d : memref<?x?xf32>, memref<?x?xf32>

    %a = memref.alloc(%I, %J) : memref<?x?xf32>
    // MLIR may actually ensure that a freshlly alloced memref is
    scf.for %i = %c0 to %I step %c1 {
        scf.for %j = %c0 to %J step %c1 {
            memref.store %i0, %a[%i, %j] : memref<?x?xf32>
        }
    }
    %d_a = gpu.alloc(%I, %J) : memref<?x?xf32>
    gpu.memcpy %d_a, %a : memref<?x?xf32>, memref<?x?xf32>

    // Call kernel.
    call @sparse_mttkrp(%nnz, %J,
                        %b_coord_0, %b_coord_1,
                        %b_coord_2, %b_values,
                        %d_c, %d_d, %d_a) : (index, index,
                                                memref<?xindex>, memref<?xindex>,
                                                memref<?xindex>, memref<?xf32>,
                                                memref<?x?xf32>, memref<?x?xf32>,
                                                memref<?x?xf32>) -> ()

    // copy memory back onto CPU
    gpu.memcpy %a, %d_a : memref<?x?xf32>, memref<?x?xf32>

    // Expected output from  mttkrp_b.tns:
    // CHECK:      {{\[}}[16075,   21930,   28505,   35800,   43815],
    // CHECK-NEXT: [10000,   14225,   19180,   24865,   31280]]
    %unranked_a = memref.cast %a : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked_a) : (memref<*xf32>) -> ()

    return
}
