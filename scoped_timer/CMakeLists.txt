find_package(Corrade CONFIG REQUIRED)

add_library(scoped_timer STATIC ${CMAKE_CURRENT_SOURCE_DIR}/include/scoped_timer/scoped_timer.cpp)
target_include_directories(scoped_timer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(scoped_timer PUBLIC Corrade::Containers)
set_property(TARGET scoped_timer PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(scoped_timer::scoped_timer ALIAS scoped_timer)