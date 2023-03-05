module {
    func.func private @milliTime() -> i64

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

    func.func public @sparse_mttkrp(%NNZ : index, %J : index, %b_coord_0 : memref<?xindex>, %b_coord_1 : memref<?xindex>,
                                    %b_coord_2 : memref<?xindex>, %b_values : memref<?xf32>, %c: memref<?x?xf32>,
                                    %d: memref<?x?xf32>, %a: memref<?x?xf32>) -> (i64) attributes {llvm.emit_c_interface} {

        %start = func.call @milliTime() : () -> (i64)
        "spf.computation"() ({
            // for(int z = 0; z < NNZ; z++) {
            //   i=UFi(z);
            //   k=UFk(z);
            //   l=UFl(z);
            //   val=UFval(z);
            //   for (int j = 0; j < J; j++) {
            //     A[i,j] += val*C[k,j]*D[l,j];
            //   }
            // }
            "spf.bar"(%NNZ, %J, %b_coord_0, %b_coord_1, %b_coord_2, %b_values, %c, %d, %a) ({
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
        %stop = func.call @milliTime() : () -> (i64)
        %time = arith.subi %stop, %start: i64

        return %time : i64
    }
}