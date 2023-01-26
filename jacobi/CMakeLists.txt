add_executable(jacobi-c-example jacobi.c)
target_compile_options(jacobi-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)

add_executable(jacobi-iegenlib-c-example jacobi_IEGEN.c)
target_compile_options(jacobi-iegenlib-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-iegenlib-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)

add_executable(jacobi-no-transform-c-print-example jacobi_no_transform_statement_printer.c)
target_compile_options(jacobi-no-transform-c-print-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-no-transform-c-print-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)

add_executable(jacobi-transformed-c-print-example jacobi_transformed_statement_printer.c)
target_compile_options(jacobi-transformed-c-print-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-transformed-c-print-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)

add_executable(jacobi-expected jacobi_expected.cpp)
target_compile_options(jacobi-expected PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-expected PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_libraries(jacobi-expected
        mlir_c_runner_utils
        mlir_runner_utils
)
