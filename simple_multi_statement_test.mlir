
module @my_module {
  func.func private @printStatementCalls(memref<*xindex>) attributes {llvm.emit_c_interface}

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index

		// index
    %i = memref.alloc() : memref<1xindex>
    memref.store %c0, %i[%c0] : memref<1xindex>

		// store
    %store = memref.alloc() : memref<6x3xindex>

		// // 0
   	// %i0 = memref.load %i[%c0] : memref<1xindex>
		// memref.store %i0, %store[%i0, %c0] : memref<3x3xindex>
		// memref.store %i0, %store[%i0, %c1] : memref<3x3xindex>
		// memref.store %i0, %store[%i0, %c2] : memref<3x3xindex>
		// %i1n = arith.addi %i0, %c1 :index
   	// memref.store %i1n, %i[%c0] : memref<1xindex>

		// // 1
   	// %i1 = memref.load %i[%c0] : memref<1xindex>
		// memref.store %i1, %store[%i1, %c0] : memref<3x3xindex>
		// memref.store %i1, %store[%i1, %c1] : memref<3x3xindex>
		// memref.store %i1, %store[%i1, %c2] : memref<3x3xindex>
		// %i2n = arith.addi %i1, %c1 :index
   	// memref.store %i2n, %i[%c0] : memref<1xindex>

		// // 2
   	// %i2 = memref.load %i[%c0] : memref<1xindex>
		// memref.store %i2, %store[%i2, %c0] : memref<3x3xindex>
		// memref.store %i2, %store[%i2, %c1] : memref<3x3xindex>
		// memref.store %i2, %store[%i2, %c2] : memref<3x3xindex>
		// %i3n = arith.addi %i2, %c1 :index
   	// memref.store %i3n, %i[%c0] : memref<1xindex>

		// scf.for %arg0 = %c0 to %c3 step %c1 {
   	// 	%iold = memref.load %i[%c0] : memref<1xindex>
		//   memref.store %arg0, %store[%iold, %c0] : memref<3x3xindex>
		//   memref.store %arg0, %store[%iold, %c1] : memref<3x3xindex>
		//   memref.store %arg0, %store[%iold, %c2] : memref<3x3xindex>

		// 	%inew = arith.addi %iold, %c1 :index
   	// 	memref.store %inew, %i[%c0] : memref<1xindex>
		// }

    %T = arith.constant 2 : index
    %X = arith.constant 4 : index

		// t_vals
    %t_vals = memref.alloc() : memref<2xindex>
		memref.store %c0, %t_vals[%c0] : memref<2xindex>
		memref.store %c1, %t_vals[%c1] : memref<2xindex>

		// x_vals
    %x_vals = memref.alloc() : memref<4xindex>
		memref.store %c0, %x_vals[%c0] : memref<4xindex>
		memref.store %c1, %x_vals[%c1] : memref<4xindex>
		memref.store %c2, %x_vals[%c2] : memref<4xindex>
		memref.store %c3, %x_vals[%c3] : memref<4xindex>

    "standalone.computation"() ({
			"standalone.bar"(%T, %X, %t_vals, %x_vals) ({
			^bb0(%tt : index, %xx : index):
			%c68 = arith.constant 68 : index

			// current index
    	%iold = memref.load %i[%c0] : memref<1xindex>

			// store current values
		  memref.store %c0, %store[%iold, %c0] : memref<6x3xindex>
		  memref.store %tt, %store[%iold, %c1] : memref<6x3xindex>
		  memref.store %xx, %store[%iold, %c2] : memref<6x3xindex>

			// increment index
			%inew = arith.addi %iold, %c1 :index
   		memref.store %inew, %i[%c0] : memref<1xindex>
			"standalone.yield"(): () -> ()
			})  {
            	reads = [
                affine_map<(t, x) -> (t)>,
                affine_map<(t, x) -> (x)>
							],
							writes = [],
            	// symbols,ufInputs,inputs,outputs
							operand_segment_sizes = dense<[2,0,2,0]> : vector<4xi32>,
							ufNames = [],
							symbolNames = ["T", "X"],
							iteratorTypes = ["reduction", "reduction"],
							executionSchedule = "{[t,x]->[t,0,x,0]}",
							iterationSpace = "{[t,x]: 1<=t<=T and 1<=x<=X}"
					} : (index, index, memref<2xindex>, memref<4xindex>) -> ()

			"standalone.bar"(%T, %X, %t_vals, %x_vals) ({
			^bb0(%tt : index, %xx : index):
			%c68 = arith.constant 68 : index

			// current index
    	%iold = memref.load %i[%c0] : memref<1xindex>

			// store current values
		  memref.store %c1, %store[%iold, %c0] : memref<6x3xindex>
		  memref.store %tt, %store[%iold, %c1] : memref<6x3xindex>
		  memref.store %xx, %store[%iold, %c2] : memref<6x3xindex>

			// increment index
			%inew = arith.addi %iold, %c1 :index
   		memref.store %inew, %i[%c0] : memref<1xindex>
			"standalone.yield"(): () -> ()
			})  {
            	reads = [
                affine_map<(t, x) -> (t)>,
                affine_map<(t, x) -> (x)>
							],
							writes = [],
            	// symbols,ufInputs,inputs,outputs
							operand_segment_sizes = dense<[2,0,2,0]> : vector<4xi32>,
							ufNames = [],
							symbolNames = ["T", "X"],
							iteratorTypes = ["reduction", "reduction"],
							executionSchedule = "{[t,x]->[t,1,x,0]}",
							iterationSpace = "{[t,x]: 1<=t<=T and 1<=x<=X}"
					} : (index, index, memref<2xindex>, memref<4xindex>) -> ()
		}): () -> ()

    // %cast_i = memref.cast %i : memref<1xindex> to memref<*xindex>
		// call @printS0(%cast_i): (memref<*xindex>) -> ()

    %cast_store = memref.cast %store : memref<6x3xindex> to memref<*xindex>
		call @printStatementCalls(%cast_store): (memref<*xindex>) -> ()

		return
	}
}