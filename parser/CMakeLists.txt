add_executable(parser main.cpp)
target_compile_options(parser PRIVATE -fno-omit-frame-pointer -fsanitize=address)
target_link_options(parser PRIVATE -fno-omit-frame-pointer -fsanitize=address)

target_link_libraries(parser
        LLVMSupport
        )