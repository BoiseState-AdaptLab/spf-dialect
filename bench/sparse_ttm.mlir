module {
    func.func private @milliTime() -> i64

    func.func private @UFfptr(%fptr: memref<?xindex>, %b_coord_constant : memref<?xindex>, %f :index) -> (index) {
        %m = memref.load %fptr[%f] : memref<?xindex>
        return %m : index
    }

    func.func private @UFxCoordConstant (%fptr: memref<?xindex>, %b_coord_constant : memref<?xindex>, %m :index) -> (index) {
        %k = memref.load %b_coord_constant[%m] : memref<?xindex>
        return %k : index
    }

    func.func public @sparse_ttm(%Mf : index, %R : index, %fptr: memref<?xindex>, %x_coord_constant : memref<?xindex>,
                                 %x_values : memref<?xf32>, %u: memref<?x?xf32>,
                                 %y: memref<?x?xf32>) -> (i64) attributes {llvm.emit_c_interface} {

        %start = func.call @milliTime() : () -> (i64)
        "spf.computation"() ({
            // for(int f = 0; f < Mf; f++) {
            //     for(int m = UFfptr(f); m < UFfptr(f+1); m++) {
            //         int k = x_coord_constant[m];
            //         for(sptIndex r = 0; r < R; ++r) {
            //             y[f,r] += x_values[m] * u[k,r];
            //         }
            //     }
            // }
            "spf.bar"(%Mf, %R, %fptr, %x_coord_constant, %x_values, %u, %y) ({
                ^bb0(%x_value : f32, %u_k_r : f32, %y_f_r : f32):
                %0 = arith.mulf %x_value, %u_k_r : f32
                %1 = arith.addf %0, %y_f_r : f32
                "spf.yield"(%1) : (f32) -> ()
            }) {
                reads = [
                    [affine_map<(f, m, k, r) -> (m)>],
                    [affine_map<(f, m, k, r) -> (k, r)>]
                ],
                writes = [
                    [affine_map<(f, m, k, r) -> (f, r)>]
                ],
                // symbols,ufInputs,inputs,outputs
                operand_segment_sizes = array<i32: 2,2,2,1>,
                symbolNames = ["Mf", "R"],
                iteratorTypes = ["parallel", "reduction", "recution", "reduction"],
                iterationSpace = "{[f,m,k,r] : 0<=f<Mf and UFfptr(f)<=m<UFfptr(f+1) and k=UFxCoordConstant(m) and 0<=r<R}",
                executionSchedule = "{[f,m,k,r]->[f,m,k,r]}",
                transforms = []
            } : (index, index, memref<?xindex>, memref<?xindex>,
                memref<?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
        }) : () -> ()
        %stop = func.call @milliTime() : () -> (i64)
        %time = arith.subi %stop, %start: i64

        return %time : i64
    }
}