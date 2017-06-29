cmake_minimum_required(VERSION 2.8)
project(YGZ-Stereo)

# uncommont this to use release
#set( CMAKE_BUILD_TYPE "Release" )
set(CMAKE_BUILD_TYPE "Debug")

# make sure we use Release and warn otherwise
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()


if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -Wall -march=native -Wno-reorder -pthread")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -pthread")
else ()
    message(STATUS "Building with build type '${CMAKE_BUILD_TYPE}', turn on the compiler optimization")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -Wall -march=native -Wno-reorder -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -pthread")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

# arm and intel cpu configurations
IF (DEFINED ENV{ARM_ARCHITECTURE})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE ()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -march=native")
ENDIF ()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules)

# thirdparty libs 
find_package(OpenCV 3.0 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

set(G2O_ROOT "/home/bobin/code/libraries/g2o")
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Glog REQUIRED)
#find_package(G2O REQUIRED)

MESSAGE(STATUS "G2O includes${G2O_INCLUDE_DIR}")

# to show 'include' in QtCreator
FILE(GLOB_RECURSE INC_ALL "*/*.h")
add_custom_target(headers SOURCES ${INC_ALL})
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/Thirdparty/fast/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
        "/home/bobin/code/libraries/g2o"
        ${PROJECT_SOURCE_DIR}/Thirdparty/lsd/include/
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
)


# thirdparty
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/fast/build/libfast.so
#        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        /home/bobin/code/libraries/g2o/lib/libg2o_types_slam3d.so
        /home/bobin/code/libraries/g2o/lib/libg2o_core.so
        /home/bobin/code/libraries/g2o/lib/libg2o_stuff.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/lsd/build/liblsd.so
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${GLOG_LIBRARY}
        )

# subdirectories
include_directories(
        util/include
        cv/include
        common/include
        backend/include
        system/include
)


add_subdirectory(util)

add_subdirectory(common)

add_subdirectory(cv)

add_subdirectory(backend)

add_subdirectory(system)

set(YGZ_LIBS ygz-common ygz-cv ygz-util ygz-backend)

# build tests
add_subdirectory(test)

# build examples
