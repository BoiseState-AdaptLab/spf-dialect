func.func private @UFi(%uf_argb_coord_0 : memref<?xindex>,
                %uf_argb_coord_1 : memref<?xindex>,
                %uf_argb_coord_2 : memref<?xindex>,
                %j :index,
                %z: index)-> index {

    %i = memref.load %uf_argb_coord_0[%z] : memref<?xindex>
    return %i : index
}

func.func private @UFk(%uf_argb_coord_0 : memref<?xindex>,
                %uf_argb_coord_1 : memref<?xindex>,
                %uf_argb_coord_2 : memref<?xindex>,
                %j : index,
                %z : index) -> index {

    %k = memref.load %uf_argb_coord_1[%z] : memref<?xindex>
    return %k : index
}

func.func private @UFl(%uf_argb_coord_0 : memref<?xindex>,
                %uf_argb_coord_1 : memref<?xindex>,
                %uf_argb_coord_2 : memref<?xindex>,
                %j : index,
                %z : index) -> index {

    %l = memref.load %uf_argb_coord_2[%z] : memref<?xindex>
    return %l : index
}

func.func public @sparse_mttkrp_extra_stuff(%NNZ : index,
                                            %I: index,
                                            %J: index,
                                            %K: index,
                                            %L: index,
                                            %argb_coord_0 : memref<?xindex>,
                                            %argb_coord_1 : memref<?xindex>,
                                            %argb_coord_2 : memref<?xindex>,
                                            %argb_values : memref<?xf64>,
                                            %argc: memref<?x?xf64>,
                                            %argd: memref<?x?xf64>,
                                            %arga: memref<?x?xf64>) -> () {

    "standalone.computation"() ({
        // for(int z = 0; z < NNZ; z++) {
        //   i=UFi(z);
        //   k=UFk(z);
        //   l=UFl(z);
        //   val=UFval(z);
        //   for (int j = 0; j < J; j++)
        //     A[i,j] += val*D[l,j]*C[k,j];
        // }
        "standalone.bar"(%NNZ, %I, %J, %K, %L, %argb_coord_0, %argb_coord_1, %argb_coord_2, %argb_values, %argc, %argd, %arga) ({
        ^bb0(%b_i_k_l : f64, %c_k_j : f64, %d_l_j : f64, %a_i_j : f64):
        %0 = arith.mulf %b_i_k_l, %d_l_j : f64
        %1 = arith.mulf %0, %c_k_j : f64
        %2 = arith.addf %1, %a_i_j : f64
        "standalone.yield"(%2) : (f64) -> ()
        }) {
               reads = [
                   affine_map<(j, z, i, k, l) -> (z)>,
                   affine_map<(j, z, i, k, l) -> (k, j)>,
                   affine_map<(j, z, i, k, l) -> (l, j)>
               ],
               writes = [
                   affine_map<(j, z, i, k, l) -> (i, j)>
               ],
               // symbols,ufInputs,inputs,outputs
               operand_segment_sizes = dense<[5,3,3,1]> : vector<4xi32>,
               ufNames = ["UFi", "UFk", "UFl"],
               symbolNames = ["NNZ", "I", "J", "K", "L"],
               iteratorTypes = ["reduction", "reduction", "reduction", "reduction", "reduction"],
               executionSchedule = "{[j,z,i,k,l]->[j,z,i,k,l]}",
               iterationSpace = "{[j,z,i,k,l]: 0<=j<J and 0<=z<NNZ and i=UFi(z) and k=UFk(z) and l=UFl(z)}",
               transforms = []
           } : (index, index, index, index, index,
               memref<?xindex>, memref<?xindex>,
               memref<?xindex>, memref<?xf64>,
               memref<?x?xf64>, memref<?x?xf64>,
               memref<?x?xf64>) -> ()
    }) : () -> ()

    return
}
