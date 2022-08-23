module {
  func.func private @printMemrefShapeF32(memref<*xf32>) attributes { llvm.emit_c_interface }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Initialize [2,3] matrix.
    %matrix = memref.alloc() : memref<2x3xf32>
    %m_dim_x = memref.dim %matrix, %c0 : memref<2x3xf32>
    %m_dim_y = memref.dim %matrix, %c1 : memref<2x3xf32>
    scf.for %i = %c0 to %m_dim_x step %c1 {
      scf.for %j = %c0 to %m_dim_y step %c1 {
        %prod = arith.muli %i,  %m_dim_y : index
        %val = arith.addi %prod, %j : index
        %val_i64 = arith.index_cast %val : index to i64
        %val_f32 = arith.sitofp %val_i64 : i64 to f32
        memref.store %val_f32, %matrix[%i, %j] : memref<2x3xf32>
      }
    }
    %matrix_unranked = memref.cast %matrix : memref<2x3xf32> to memref<*xf32>
    call @printMemrefShapeF32(%matrix_unranked) : (memref<*xf32>) -> ()

    // Initialize [3] vector.
    %vector = memref.alloc() : memref<3xf32>
    %v_dim_x = memref.dim %vector, %c0 : memref<3xf32>
    scf.for %i = %c0 to %v_dim_x step %c1 {
      %val_i64 = arith.index_cast %i : index to i64
      %val_f32 = arith.sitofp %val_i64 : i64 to f32
      memref.store %val_f32, %vector[%i] : memref<3xf32>
    }
    %vector_unranked = memref.cast %vector : memref<3xf32> to memref<*xf32>
    call @printMemrefShapeF32(%vector_unranked) : (memref<*xf32>) -> ()


    %out = memref.alloc() : memref<2xf32>
    linalg.matvec ins(%matrix, %vector : memref<2x3xf32>, memref<3xf32>) outs(%out : memref<2xf32>)

    %out_unranked = memref.cast %out : memref<2xf32> to memref<*xf32>
    call @printMemrefShapeF32(%out_unranked) : (memref<*xf32>) -> ()

    memref.dealloc %matrix : memref<2x3xf32>
    memref.dealloc %vector : memref<3xf32>
    memref.dealloc %out : memref<2xf32>
    return
  }
}