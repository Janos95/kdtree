cmake_minimum_required(VERSION 3.16)
project(kdtree)

set(CMAKE_CXX_STANDARD 17)

find_package(Magnum CONFIG REQUIRED)

add_subdirectory(external/nanoflann)
add_subdirectory(scoped_timer)

add_executable(kdtree kdtree.h main.cpp)
target_link_libraries(kdtree
        Magnum::Magnum
        nanoflann::nanoflann
        scoped_timer::scoped_timer)
