module {
    func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

    func.func @dense_mttkrp(
                   %argb: memref<?x?x?xf64>,
                   %argc: memref<?x?xf64>,
                   %argd: memref<?x?xf64>,
                   %arga: memref<?x?xf64>) -> memref<?x?xf64> {

        "standalone.bar"(%argb, %argc, %argd, %arga) ({
        ^bb0(%b_i_k_l : f64, %c_k_j : f64, %d_l_j : f64, %a_i_j : f64):
        %0 = arith.mulf %b_i_k_l, %d_l_j : f64
        %1 = arith.mulf %0, %c_k_j : f64
        %2 = arith.addf %1, %a_i_j : f64
        "standalone.yield"(%2) : (f64) -> ()
        })  {
                indexing_maps = [
                                    affine_map<(i, k, l, j) -> (i, k, l)>,
                                    affine_map<(i, k, l, j) -> (k, j)>,
                                    affine_map<(i, k, l, j) -> (l, j)>,
                                    affine_map<(i, k, l, j) -> (i, j)>
                                ],
                operand_segment_sizes = dense<[3, 1]> : vector<2xi32>
            } : (memref<?x?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()

        return %arga : memref<?x?xf64>
    }

    func.func @main() {
        // constants of float type
        %f0 = arith.constant 0.0 : f64
        %f3 = arith.constant 3.0 : f64
        %f63 = arith.constant 63.0 : f64
        %f11 = arith.constant 11.0 : f64
        %f100 = arith.constant 100.0 : f64
        %f66 = arith.constant 66.0 : f64
        %f61 = arith.constant 61.0 : f64
        %f13 = arith.constant 13.0 : f64
        %f43 = arith.constant 43.0 : f64
        %f77 = arith.constant 77.0 : f64
        %f10 = arith.constant 10.0 : f64
        %f46 = arith.constant 46.0 : f64
        %f53 = arith.constant 53.0 : f64
        %f75 = arith.constant 75.0 : f64
        %f22 = arith.constant 22.0 : f64
        %f18 = arith.constant 18.0 : f64

        // constants of index type
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index

        // Dimensions of matrices for mttkrp_b
        %I = arith.constant 2 : index
        %J = arith.constant 5 : index
        %K = arith.constant 3 : index
        %L = arith.constant 4 : index
        %nnz = arith.constant 17 : index

        // Construct dense B matrix by manually writing contentes of mttkrp_b.tns.
        // TODO: read this from file
        %b = memref.alloc(%I, %K, %L) : memref<?x?x?xf64>
        scf.for %i = %c0 to %I step %c1 {
            scf.for %k = %c0 to %K step %c1 {
                scf.for %l = %c0 to %L step %c1 {
                    memref.store %f0, %b[%i, %k, %l] : memref<?x?x?xf64>
                }
            }
        }
        memref.store %f3, %b[%c0, %c0, %c2] : memref<?x?x?xf64>
        memref.store %f63, %b[%c0, %c0, %c3] : memref<?x?x?xf64>
        memref.store %f11, %b[%c0, %c1, %c1] : memref<?x?x?xf64>
        memref.store %f100, %b[%c0, %c1, %c2] : memref<?x?x?xf64>
        memref.store %f66, %b[%c0, %c2, %c0] : memref<?x?x?xf64>
        memref.store %f61, %b[%c0, %c2, %c1] : memref<?x?x?xf64>
        memref.store %f13, %b[%c0, %c2, %c2] : memref<?x?x?xf64>
        memref.store %f43, %b[%c0, %c2, %c3] : memref<?x?x?xf64>
        memref.store %f77, %b[%c1, %c0, %c0] : memref<?x?x?xf64>
        memref.store %f10, %b[%c1, %c0, %c2] : memref<?x?x?xf64>
        memref.store %f46, %b[%c1, %c0, %c3] : memref<?x?x?xf64>
        memref.store %f61, %b[%c1, %c1, %c0] : memref<?x?x?xf64>
        memref.store %f53, %b[%c1, %c1, %c1] : memref<?x?x?xf64>
        memref.store %f3, %b[%c1, %c1, %c2] : memref<?x?x?xf64>
        memref.store %f75, %b[%c1, %c1, %c3] : memref<?x?x?xf64>
        memref.store %f22, %b[%c1, %c2, %c1] : memref<?x?x?xf64>
        memref.store %f18, %b[%c1, %c2, %c2] : memref<?x?x?xf64>
        // %unranked_b = memref.cast %b : memref<?x?x?xf64> to memref<*xf64>
        // call @printMemrefF64(%unranked_b) : (memref<*xf64>) -> ()

        // Initialize dense C and D inputs and dense output A.
        %c = memref.alloc(%K, %J) : memref<?x?xf64>
        scf.for %k = %c0 to %K step %c1 {
            scf.for %j = %c0 to %J step %c1 {
                %v0 = arith.muli %k, %J : index
                %v1 = arith.addi %v0, %j : index
                %v2 = arith.index_cast %v1 : index to i32
                %v = arith.sitofp %v2 : i32 to f64
                memref.store %v, %c[%k, %j] : memref<?x?xf64>
            }
        }
        // %unranked_c = memref.cast %c : memref<?x?xf64> to memref<*xf64>
        // call @output_memref_f64(%unranked_c) : (memref<*xf64>) -> ()

        %d = memref.alloc(%L, %J) : memref<?x?xf64>
        scf.for %l = %c0 to %L step %c1 {
            scf.for %j = %c0 to %J step %c1 {
                %v0 = arith.muli %l, %J : index
                %v1 = arith.addi %v0, %j : index
                %v2 = arith.index_cast %v1 : index to i32
                %v = arith.sitofp %v2 : i32 to f64
                memref.store %v, %d[%l, %j] : memref<?x?xf64>
            }
        }
        // %unranked_d = memref.cast %d : memref<?x?xf64> to memref<*xf64>
        // call @output_memref_f64(%unranked_d) : (memref<*xf64>) -> ()

        %a = memref.alloc(%I, %J) : memref<?x?xf64>
        scf.for %i = %c0 to %I step %c1 {
            scf.for %j = %c0 to %J step %c1 {
                memref.store %f0, %a[%i, %j] : memref<?x?xf64>
            }
        }
        // %unranked_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
        // call @output_memref_f64(%unranked_a) : (memref<*xf64>) -> ()

        // Call kernel.
        %out = call @dense_mttkrp(%b, %c, %d, %a) : (memref<?x?x?xf64>, memref<?x?xf64>,
                                                     memref<?x?xf64>, memref<?x?xf64>)
                                                     -> memref<?x?xf64>

        // Expected output from  mttkrp_b.tns:
        // ( ( 16075, 21930, 28505, 35800, 43815 ), ( 10000, 14225, 19180, 24865, 31280 ) )
        %unranked_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
        call @printMemrefF64(%unranked_a) : (memref<*xf64>) -> ()

        return
    }
}
