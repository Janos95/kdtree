cmake_minimum_required(VERSION 3.16)
project(kdtree)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

find_package(Magnum CONFIG REQUIRED COMPONENTS Sdl2Application GL SceneGraph Shaders Trade MeshTools Primitives WindowlessEglApplication)
find_package(MagnumPlugins CONFIG REQUIRED AssimpImporter)
find_package(Eigen3)

add_subdirectory(external/nanoflann)
add_subdirectory(scoped_timer)

add_executable(bench benchmark.cpp)
add_executable(app application.cpp arc_ball.cpp)

corrade_add_resource(KdTree_Rcs resources.conf)

add_library(kdtree STATIC
        kdtree.h
        kdtree_shader.h
        kdtree_shader.cpp
        ${KdTree_Rcs})

target_link_libraries(kdtree PUBLIC Magnum::Magnum Magnum::GL)

add_library(kdtree::kdtree ALIAS kdtree)

target_link_libraries(bench
        Magnum::Magnum
        Magnum::WindowlessEglApplication
        nanoflann::nanoflann
        scoped_timer::scoped_timer
        kdtree::kdtree)

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
        Eigen3::Eigen
        scoped_timer::scoped_timer
        kdtree::kdtree)
