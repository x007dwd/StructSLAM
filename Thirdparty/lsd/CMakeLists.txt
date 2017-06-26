cmake_minimum_required(VERSION 2.8)
project(lsd)

set(CMAKE_CXX_STANDARD 11)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

add_library(lsd SHARED
        src/lsd.cpp

        )

include_directories(
        include
)
add_executable(lsd_cmd
        src/lsd_cmd.cpp
        )
target_link_libraries(lsd_cmd
        lsd
        )
add_executable(lsd_call_example
        example/lsd_call_example.cpp
        )
target_link_libraries(lsd_call_example
        lsd
        )
