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
// RUN:  | mlir-cpu-runner \
// RUN:    -entry-point-result=void \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:    -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:    | FileCheck %s

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

    %ub_T = arith.constant 10 : index
    %lb_x = arith.constant 1 : index
    %ub_x = arith.constant 8 : index
    %ub_T_div_2 = arith.floordivsi %ub_T, %c2 : index

    %A = memref.alloc() : memref<10xf64>
    %B = memref.alloc() : memref<10xf64>
    scf.for %i = %c0 to %c10 step %c1 {
        memref.store %f0, %A[%i] : memref<10xf64>
        memref.store %f0, %B[%i] : memref<10xf64>
    }
    memref.store %f100, %A[%c9] : memref<10xf64>
    memref.store %f100, %B[%c9] : memref<10xf64>

    "spf.computation"() ({
        "spf.statement"(%ub_T_div_2, %lb_x, %ub_x, %B, %A) ({
        ^stmt(%B_x_plus_one: f64, %B_x: f64, %B_x_minus_one: f64):
        %0 = arith.addf %B_x_plus_one, %B_x : f64
        %1 = arith.addf %0, %B_x_minus_one : f64
        %2 = arith.divf %1, %f3 : f64
        "spf.yield"(%2): (f64) -> ()
        })  {  reads = [
                           [ // data access functions for first input
                               affine_map<(t, x) -> (x+1)>,
                               affine_map<(t, x) -> (x)>,
                               affine_map<(t, x) -> (x-1)>
                           ]
                       ],
               writes = [[affine_map<(t, x) -> (x)>]],
               // symbols, ufInputs, inputs, outputs
               operand_segment_sizes=array<i32: 3,0,1,1>,
               symbolNames= ["ub_T", "lb_x", "ub_x"],
               iteratorTypes = ["reduction", "reduction"],
               executionSchedule = "{[t,x]->[t,0,x,0]}",
               iterationSpace = "{[t,x]: 1<=t<=ub_T and lb_x<=x<=ub_x}",
               transforms = ["{[a,b,c,d]->[a,0,x,0]:x=c-1}"]
            }:(index,index,index,memref<10xf64>,memref<10xf64>)->()
        "spf.statement"(%ub_T_div_2, %lb_x, %ub_x, %A, %B) ({
        ^stmt(%A_x_plus_one: f64, %A_x: f64, %A_x_minus_one: f64):
        %0 = arith.addf %A_x_plus_one, %A_x : f64
        %1 = arith.addf %0, %A_x_minus_one : f64
        %2 = arith.divf %1, %f3 : f64
        "spf.yield"(%2): (f64) -> ()
        })  {  reads = [
                   [ // data access functions for first input
                       affine_map<(t, x) -> (x+1)>,
                       affine_map<(t, x) -> (x)>,
                       affine_map<(t, x) -> (x-1)>
                   ]
               ],
               writes = [[affine_map<(t, x) -> (x)>]],
               // symbols, ufInputs, inputs, outputs
               operand_segment_sizes = array<i32: 3,0,1,1>,
               symbolNames = ["ub_T", "lb_x", "ub_x"],
               iteratorTypes = ["reduction", "reduction"],
               executionSchedule = "{[t,x]->[t,1,x,0]}",
               iterationSpace = "{[t,x]: 1<=t<=ub_T and lb_x<=x<=ub_x}",
               transforms = ["{[a,b,c,d]->[a,0,c,1]}"]
            }:(index,index,index,memref<10xf64>,memref<10xf64>)->()
    }): () -> ()


    // CHECK: [0,  0.0558858,  0.330234,  1.35142,  4.24732,  10.8317,  23.2078,  42.8085,  69.2831,  100]
	%unranked_A = memref.cast %A : memref<10xf64> to memref<*xf64>
	call @printMemrefF64(%unranked_A) : (memref<*xf64>) -> ()

    // CHECK: [0,  0.128707,  0.57918,  1.97632,  5.47681,  12.7623,  25.616,  45.0998,  70.6972,  100]
	%unranked_B = memref.cast %B : memref<10xf64> to memref<*xf64>
	call @printMemrefF64(%unranked_B) : (memref<*xf64>) -> ()

    return
}