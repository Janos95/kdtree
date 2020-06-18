cmake_minimum_required(VERSION 3.16)
project(kdtree)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

find_package(Magnum CONFIG REQUIRED COMPONENTS Sdl2Application GL SceneGraph Shaders Trade MeshTools Primitives)
find_package(MagnumPlugins CONFIG REQUIRED AssimpImporter)

add_subdirectory(external/nanoflann)
add_subdirectory(scoped_timer)

add_executable(bench kdtree.h benchmark.cpp)
add_executable(app kdtree.h application.cpp arc_ball.cpp)

target_link_libraries(bench
        Magnum::Magnum
        nanoflann::nanoflann
        scoped_timer::scoped_timer)

target_link_libraries(app
        Magnum::Magnum
        Magnum::Application
        Magnum::GL
        Magnum::SceneGraph
        Magnum::Shaders
        Magnum::Trade
        Magnum::MeshTools
        Magnum::Primitives
        nanoflann::nanoflann
        scoped_timer::scoped_timer)
