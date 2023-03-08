// RUN: spf-opt %s \
// RUN:  -convert-spf-to-loops \
// RUN:  -inline \
// RUN:  -cse \
// RUN:  -lower-affine \
// RUN:  -convert-vector-to-scf \
// RUN:  -convert-scf-to-cf \
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
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:    | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)
func.func private @coords(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
func.func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
func.func private @values(!llvm.ptr<i8>) -> memref<?xf32> attributes {llvm.emit_c_interface}

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
        // for(int z = 0; z < NNZ; z++) {
        //   i=UFi(z);
        //   k=UFk(z);
        //   l=UFl(z);
        //   val=UFval(z);
        //   for (int j = 0; j < J; j++)
        //     A[i,j] += val*C[k,j]*D[l,j];
        // }
        "spf.statement"(%NNZ, %J, %argb_coord_0, %argb_coord_1, %argb_coord_2, %argb_values, %argc, %argd, %arga) ({
            ^bb0(%b_i_k_l : f32, %c_k_j : f32, %d_l_j : f32, %a_i_j : f32):
            %0 = arith.mulf %b_i_k_l, %d_l_j : f32
            %1 = arith.mulf %0, %c_k_j : f32
            %2 = arith.addf %1, %a_i_j : f32
            "spf.yield"(%2) : (f32) -> ()
        }) {
            reads = [
                [affine_map<(z, i, k, l, j) -> (z)>],
                [affine_map<(z, i, k, l, j) -> (k, j)>],
                [affine_map<(z, i, k, l, j) -> (l, j)>]
            ],
            writes = [
                [affine_map<(z, i, k, l, j) -> (i, j)>]
            ],
            // symbols, ufInputs, inputs, outputs
            operand_segment_sizes = array<i32: 2,3,3,1>,
            symbolNames = ["NNZ", "J"],
            iteratorTypes = ["reduction", "reduction", "reduction", "reduction", "parallel"],
            executionSchedule = "{[z,i,k,l,j]->[z,i,k,l,j]}",
            iterationSpace = "{[z,i,k,l,j]: 0<=z<NNZ and i=UFi(z) and k=UFk(z) and l=UFl(z) and 0<=j<J}",
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
    // // constants of float type
    %f0 = arith.constant 0.0 : f32

    // constants of index type
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
    %b_coord_0 = call @coords(%storage, %c0) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
    %b_coord_1 = call @coords(%storage, %c1) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
    %b_coord_2 = call @coords(%storage, %c2) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
    %b_values = call @values(%storage) : (!llvm.ptr<i8>) -> (memref<?xf32>)
    // %unranked_b_values = memref.cast %b_values : memref<?xf32> to memref<*xf32>
    // call @printMemrefF32(%unranked_b_values) : (memref<*xf32>) -> ()

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
    // %unranked_c = memref.cast %c : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%unranked_c) : (memref<*xf32>) -> ()

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
    // %unranked_d = memref.cast %d : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%unranked_d) : (memref<*xf32>) -> () //

    %a = memref.alloc(%I, %J) : memref<?x?xf32>
    scf.for %i = %c0 to %I step %c1 {
        scf.for %j = %c0 to %J step %c1 {
            memref.store %f0, %a[%i, %j] : memref<?x?xf32>
        }
    }
    // %unranked_a_before = memref.cast %a : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%unranked_a_before) : (memref<*xf32>) -> ()

    // Call kernel.
    call @sparse_mttkrp(%nnz, %J,
                        %b_coord_0, %b_coord_1,
                        %b_coord_2, %b_values,
                        %c, %d, %a) : (index, index,
                                        memref<?xindex>, memref<?xindex>,
                                        memref<?xindex>, memref<?xf32>,
                                        memref<?x?xf32>, memref<?x?xf32>,
                                        memref<?x?xf32>) -> ()

    // Expected output from  mttkrp_b.tns:
    // CHECK:      {{\[}}[16075,   21930,   28505,   35800,   43815],
    // CHECK-NEXT: [10000,   14225,   19180,   24865,   31280]]
    %unranked_a = memref.cast %a : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked_a) : (memref<*xf32>) -> ()

    return
}
