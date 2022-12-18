add_executable(jacobi-c-example jacobi.c)
add_executable(jacobi-iegenlib-c-example jacobi_IEGEN.c)

target_compile_options(jacobi-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)

target_compile_options(jacobi-iegenlib-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(jacobi-iegenlib-c-example PRIVATE -fno-omit-frame-pointer -fsanitize=address)