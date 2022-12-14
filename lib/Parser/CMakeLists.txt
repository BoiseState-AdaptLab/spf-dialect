add_executable(try_stuff_out main.cpp)
target_compile_options(try_stuff_out PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(try_stuff_out PRIVATE -fno-omit-frame-pointer -fsanitize=address)

target_link_libraries(try_stuff_out
        LLVMSupport
        )
