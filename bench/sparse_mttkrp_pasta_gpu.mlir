module {
    func.func private @milliTime() -> i64

	func.func private @UFi(%uf_argb_coord_0 : memref<?xindex>,
						   %uf_argb_coord_1 : memref<?xindex>,
						   %uf_argb_coord_2 : memref<?xindex>,
                           %THREADS_Y : index,
                           %NNZ_PER_LOOP : index,
						   %z: index)-> index {
        %i = memref.load %uf_argb_coord_0[%z] : memref<?xindex>
        return %i : index
	}

	func.func private @UFk(%uf_argb_coord_0 : memref<?xindex>,
						   %uf_argb_coord_1 : memref<?xindex>,
						   %uf_argb_coord_2 : memref<?xindex>,
                           %THREADS_Y : index,
                           %NNZ_PER_LOOP : index,
						   %z : index) -> index {
        %k = memref.load %uf_argb_coord_1[%z] : memref<?xindex>
        return %k : index
	}

	func.func private @UFl(%uf_argb_coord_0 : memref<?xindex>,
						   %uf_argb_coord_1 : memref<?xindex>,
						   %uf_argb_coord_2 : memref<?xindex>,
                           %THREADS_Y : index,
                           %NNZ_PER_LOOP : index,
						   %z : index) -> index {
        %l = memref.load %uf_argb_coord_2[%z] : memref<?xindex>
        return %l : index
	}

	func.func private @UFx(%uf_argb_coord_0 : memref<?xindex>,
						   %uf_argb_coord_1 : memref<?xindex>,
						   %uf_argb_coord_2 : memref<?xindex>,
                           %THREADS_NNZ : index,
                           %NNZ_PER_LOOP : index,
						   %block : index,
						   %ty : index,
						   %nl : index) -> index {
        //x = blockIdx.x * blokDim.y + tidy + nl * nnz_per_loop
        %block_times_THREADS_NNZ = arith.muli %block, %THREADS_NNZ : index
        %nl_times_NNZ_PER_LOOP = arith.muli %nl, %NNZ_PER_LOOP : index
        %block_times_THREADS_NNZ_plus_nl_times_NNZ_PER_LOOP = arith.addi %block_times_THREADS_NNZ, %nl_times_NNZ_PER_LOOP : index
        %x = arith.addi %block_times_THREADS_NNZ_plus_nl_times_NNZ_PER_LOOP, %ty : index
		return %x : index
	}

	func.func @sparse_mttkrp(%NNZ : index,
                             %J: index,
                             %argb_coord_0 : memref<?xindex>,
                             %argb_coord_1 : memref<?xindex>,
                             %argb_coord_2 : memref<?xindex>,
                             %argb_values : memref<?xf32>,
                             %argc: memref<?x?xf32>,
                             %argd: memref<?x?xf32>,
                             %arga: memref<?x?xf32>) -> (i64) attributes {llvm.emit_c_interface} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // I have no idea why pasta chose this number
        %BLOCKS = arith.constant 32768 : index
        // This implementation doesn't tile J in the same way pasta does for
        // larger values of J than our benchmarks use it could compute the wrong
        // answer.
        %THREADS_J = arith.addi %J, %c0 : index
        %THREADS_NNZ = arith.constant 51 : index
        %NNZ_PER_LOOP = arith.muli %BLOCKS, %THREADS_NNZ : index
        // this is all to create celing division. There is a ceildiv operator
        // but I had some issues with it not lowering.
        %tmp0 = arith.addi %NNZ, %NNZ_PER_LOOP : index
        %tmp1 = arith.subi %tmp0, %c1 : index
        %NUM_LOOPS_NNZ = arith.divui %tmp1, %NNZ_PER_LOOP : index

        %start = func.call @milliTime() : () -> (i64)
        "standalone.computation"() ({
            // MTTKRP for GPUs: (this is a stright port of pasta implementation)
            //
            // NNZ_PER_LOOP = BLOCKS * TRHEADS_NNZ
            // NUM_LOOPS_NNZ = (nnz + NNZ_PER_LOOP - 1) / NNZ_PER_LOOP
            //
            // for (int block = 0; block < BLOCKS; block++) {
            //   for (int j = 0; j < THREADS_J; j++) {
            //     for(int tnnz = 0; tnnz < THREADS_NNZ; tnnz++) {
            //       for(int nl = 0; nl < NUM_LOOPS_NNZ; nl++) {
            //          x = block * THREADS_NNZ + tnnz + nl * NNZ_PER_LOOP
            //          if x < nnz {
            //              i=UFi(z);
            //              k=UFk(z);
            //              l=UFl(z);
            //              val=UFval(z);
            //              A[i,j] += val*D[l,j]*C[k,j];
            //          }
            //       }
            //     }
            //   }
            // }
            "standalone.bar"(%NNZ, %BLOCKS, %THREADS_J, %THREADS_NNZ, %NUM_LOOPS_NNZ, %J, // symbols
                             %argb_coord_0, %argb_coord_1, %argb_coord_2, %THREADS_NNZ, %NNZ_PER_LOOP, // ufInputs
                             %argb_values, %argc, %argd, // inputs
                             %arga) ({ // outputs
            ^bb0(%b_i_k_l : f32, %c_k_j : f32, %d_l_j : f32, %a_i_j : f32):
            %0 = arith.mulf %b_i_k_l, %d_l_j : f32
            %1 = arith.mulf %0, %c_k_j : f32
            %2 = arith.addf %1, %a_i_j : f32
            "standalone.yield"(%2) : (f32) -> ()
            })  {
                    reads = [
                        [affine_map<(block, j, tnnz, nl, x, i, k, l) -> (x)>],
                        [affine_map<(block, j, tnnz, nl, x, i, k, l) -> (k, j)>],
                        [affine_map<(block, j, tnnz, nl, x, i, k, l) -> (l, j)>]
                    ],
                    writes = [
                        [affine_map<(block, j, tnnz, nl, x, i, k, l) -> (i, j)>]
                    ],
                    // symbols,ufInputs,inputs,outputs
                    operand_segment_sizes = array<i32: 6,5,3,1>,
                    symbolNames = ["NNZ", "BLOCKS", "THREADS_J", "THREADS_NNZ", "NUM_LOOPS_NNZ", "J"],
                    iteratorTypes = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction", "reduction"],
                    executionSchedule = "{[block,j,tnnz,nl,x,i,k,l]->[block,j,tnnz,nl,x,i,k,l]}",
                    iterationSpace = "{[block,j,tnnz,nl,x,i,k,l] : 0<=block<BLOCKS and 0<=j<THREADS_J and 0<=tnnz<THREADS_NNZ and 0<=nl<NUM_LOOPS_NNZ and x=UFx(block,tnnz,nl) and x<NNZ and i=UFi(x) and k=UFk(x) and l=UFl(x)}",
                    transforms = []
                } : (index, index, index, index, index, index,
                     memref<?xindex>, memref<?xindex>,
                     memref<?xindex>, index, index, memref<?xf32>,
                     memref<?x?xf32>, memref<?x?xf32>,
                     memref<?x?xf32>) -> ()
        }) : () -> ()
        %stop = func.call @milliTime() : () -> (i64)
        %time = arith.subi %stop, %start: i64

        %bla = arith.constant 69 : i64
        return %bla : i64
	}
}
