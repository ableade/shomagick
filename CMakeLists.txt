cmake_minimum_required (VERSION 2.6)
project (Shomagick)

set(CMAKE_CXX_FLAGS "-std=c++17")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${SOURCE_DIR}/cmake/Modules/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories( ${INCLUDE_DIR} )

set(Shomagick_VERSION_MAJOR 1)
set(Shomagick_VERSION_MINOR 0)

configure_file (
  "${SOURCE_DIR}/ShomagickConfig.h.in"
  "${PROJECT_BINARY_DIR}/ShomagickConfig.h"
)

include_directories("${PROJECT_BINARY_DIR}")
add_subdirectory(src)