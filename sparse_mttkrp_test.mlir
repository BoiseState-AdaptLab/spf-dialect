module {
	func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
	func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)
	func.func private @coords(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
	func.func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
	func.func private @values(!llvm.ptr<i8>) -> memref<?xf64> attributes {llvm.emit_c_interface}
	func.func private @rtclock() -> f64

	func.func @UFi(%uf_argb_coord_0 : memref<?xindex>,
									%uf_argb_coord_1 : memref<?xindex>,
									%uf_argb_coord_2 : memref<?xindex>,
									%j :index,
									%z: index)-> index {

			%i = memref.load %uf_argb_coord_0[%z] : memref<?xindex>
			return %i : index
	}

	func.func @UFk(%uf_argb_coord_0 : memref<?xindex>,
									%uf_argb_coord_1 : memref<?xindex>,
									%uf_argb_coord_2 : memref<?xindex>,
									%j : index,
									%z : index) -> index {

			%k = memref.load %uf_argb_coord_1[%z] : memref<?xindex>
			return %k : index
	}

	func.func @UFl(%uf_argb_coord_0 : memref<?xindex>,
									%uf_argb_coord_1 : memref<?xindex>,
									%uf_argb_coord_2 : memref<?xindex>,
									%j : index,
									%z : index) -> index {

			%l = memref.load %uf_argb_coord_2[%z] : memref<?xindex>
			return %l : index
	}

	func.func @sparse_mttkrp(%NNZ : index,
													 %J: index,
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
					"standalone.bar"(%NNZ, %J, %argb_coord_0, %argb_coord_1, %argb_coord_2, %argb_values, %argc, %argd, %arga) ({
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
								 operand_segment_sizes = array<i32: 2,3,3,1>,
								 ufNames = ["UFi", "UFk", "UFl"],
								 symbolNames = ["NNZ", "J"],
								 iteratorTypes = ["parallel", "reduction", "reduction", "reduction", "reduction"],
								 executionSchedule = "{[j,z,i,k,l]->[j,z,i,k,l]}",
								 iterationSpace = "{[j,z,i,k,l]: 0<=j<J and 0<=z<NNZ and i=UFi(z) and k=UFk(z) and l=UFl(z)}",
								 transforms = []
						 } : (index, index,
								 memref<?xindex>, memref<?xindex>,
								 memref<?xindex>, memref<?xf64>,
								 memref<?x?xf64>, memref<?x?xf64>,
								 memref<?x?xf64>) -> ()
			}) : () -> ()

			return
	}

	func.func @main() {
			// // constants of float type
			%f0 = arith.constant 0.0 : f64

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
			%b_values = call @values(%storage) : (!llvm.ptr<i8>) -> (memref<?xf64>)
			// %unranked_b_values = memref.cast %b_values : memref<?xf64> to memref<*xf64>
			// call @printMemrefF64(%unranked_b_values) : (memref<*xf64>) -> ()

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
			// call @printMemrefF64(%unranked_c) : (memref<*xf64>) -> ()

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
			// call @printMemrefF64(%unranked_d) : (memref<*xf64>) -> () //

			%a = memref.alloc(%I, %J) : memref<?x?xf64>
			scf.for %i = %c0 to %I step %c1 {
					scf.for %j = %c0 to %J step %c1 {
							memref.store %f0, %a[%i, %j] : memref<?x?xf64>
					}
			}
			// %unranked_a_before = memref.cast %a : memref<?x?xf64> to memref<*xf64>
			// call @printMemrefF64(%unranked_a_before) : (memref<*xf64>) -> ()

			%t0 = call @rtclock() : () -> f64

			// Call kernel.
			call @sparse_mttkrp(%nnz, %J,
													%b_coord_0, %b_coord_1,
													%b_coord_2, %b_values,
													%c, %d, %a) : (index, index,
																				 memref<?xindex>, memref<?xindex>,
																				 memref<?xindex>, memref<?xf64>,
																				 memref<?x?xf64>, memref<?x?xf64>,
																				 memref<?x?xf64>) -> ()

			%t1 = call @rtclock() : () -> f64
			%t1024 = arith.subf %t1, %t0 : f64

			// Print timings.
			vector.print %t1024 : f64

			// Expected output from  mttkrp_b.tns:
			// ( ( 16075, 21930, 28505, 35800, 43815 ), ( 10000, 14225, 19180, 24865, 31280 ) )
			%unranked_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
			call @printMemrefF64(%unranked_a) : (memref<*xf64>) -> ()

			return
	}
}
