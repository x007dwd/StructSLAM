cmake_minimum_required(VERSION 2.8)
project(lsd)

set(CMAKE_CXX_STANDARD 11)


add_library(lsd_lib
        src/lsd.cpp

        )

include_directories(
        include
)
add_executable(lsd_cmd
        src/lsd_cmd.cpp
        )
target_link_libraries(lsd_cmd
        lsd_lib
        )
add_executable(lsd_call_example
        example/lsd_call_example.cpp
        )
target_link_libraries(lsd_call_example
        lsd_lib
        )
