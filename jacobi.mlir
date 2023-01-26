
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
    %f0 = arith.constant 0.0 : f64
    %f3 = arith.constant 3.0 : f64
    %f100 = arith.constant 100.0 : f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index

    %T = arith.constant 10 : index
    %X = arith.constant 8 : index
    %T_div_2 = arith.floordivsi %T, %c2 : index

    %A = memref.alloc() : memref<10xf64>
    %B = memref.alloc() : memref<10xf64>
    scf.for %i = %c0 to %c10 step %c1 {
        memref.store %f0, %A[%i] : memref<10xf64>
        memref.store %f0, %B[%i] : memref<10xf64>
    }
    memref.store %f100, %A[%c9] : memref<10xf64>
    memref.store %f100, %B[%c9] : memref<10xf64>

    "standalone.computation"() ({
        "standalone.bar"(%T_div_2, %X, %B, %A) ({
        ^bb0(%B_x_plus_one: f64, %B_x: f64, %B_x_minus_one: f64):
        %0 = arith.addf %B_x_plus_one, %B_x : f64
        %1 = arith.addf %0, %B_x_minus_one : f64
        %2 = arith.divf %1, %f3 : f64
        "standalone.yield"(%2): (f64) -> ()
        })  {
                reads = [
                    [
                        affine_map<(t, x) -> (x+1)>,
                        affine_map<(t, x) -> (x)>,
                        affine_map<(t, x) -> (x-1)>
                    ]
                ],
                writes = [
                    [
                        affine_map<(t, x) -> (x)>
                    ]
                ],
                // symbols, ufInputs, inputs, outputs
                operand_segment_sizes = array<i32: 2,0,1,1>,
                symbolNames = ["T", "X"],
                iteratorTypes = ["reduction", "reduction"],
                executionSchedule = "{[t,x]->[t,0,x]}",
                iterationSpace = "{[t,x]: 1<=t<=T and 1<=x<=X}",
                transforms = []
            } : (index, index, memref<10xf64>, memref<10xf64>) -> ()

        "standalone.bar"(%T, %X, %A, %B) ({
        ^bb0(%A_x_plus_one: f64, %A_x: f64, %A_x_minus_one: f64):
        %0 = arith.addf %A_x_plus_one, %A_x : f64
        %1 = arith.addf %0, %A_x_minus_one : f64
        %2 = arith.divf %1, %f3 : f64
        "standalone.yield"(%2): (f64) -> ()
        })  {
                reads = [
                    [ // data access functions for first input
                        affine_map<(t, x) -> (x+1)>,
                        affine_map<(t, x) -> (x)>,
                        affine_map<(t, x) -> (x-1)>
                    ]
                ],
                writes = [
                    [
                        affine_map<(t, x) -> (x)>
                    ]
                ],
                // symbols, ufInputs, inputs, outputs
                operand_segment_sizes = array<i32: 2,0,1,1>,
                symbolNames = ["T", "X"],
                iteratorTypes = ["reduction", "reduction"],
                executionSchedule = "{[t,x]->[t,1,x]}",
                iterationSpace = "{[t,x]: 1<=t<=T and 1<=x<=X}",
                transforms = []
            } : (index, index, memref<10xf64>, memref<10xf64>) -> ()
    }): () -> ()

	%unranked_A = memref.cast %A : memref<10xf64> to memref<*xf64>
	call @printMemrefF64(%unranked_A) : (memref<*xf64>) -> ()

	%unranked_B = memref.cast %B : memref<10xf64> to memref<*xf64>
	call @printMemrefF64(%unranked_B) : (memref<*xf64>) -> ()

    return
}