cmake_minimum_required(VERSION 3.13)
project(shomagick)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)
include_directories(tests)
include_directories(tests/x64)
include_directories(tests/x64/Debug)

add_executable(shomagick
        src/allclose.cpp
        src/allclose.h
        src/bootstrap.h
        src/bundle.h
        src/camera.cpp
        src/camera.h
        src/csv.h
        src/detector.h
        src/flightsession.cpp
        src/flightsession.h
        src/image.hpp
        src/json.hpp
        src/kdtree.cpp
        src/kdtree.h
        src/multiview.cpp
        src/multiview.h
        src/parser.cpp
        src/reconstruction.cpp
        src/reconstruction.h
        src/reconstructor.cpp
        src/reconstructor.h
        src/RobustMatcher.cpp
        src/RobustMatcher.h
        src/shomatcher.cpp
        src/shomatcher.hpp
        src/shot.h
        src/shotracking.cpp
        src/shotracking.h
        src/stitch.cpp
        src/types.h
        src/unionfind.h
        src/utilities.h
        tests/x64/Debug/tests.pch
        tests/allclose_tests.cpp
        tests/camera_tests.cpp
        tests/stdafx.cpp
        tests/stdafx.h
        tests/targetver.h
        tests/triangulation_tests.cpp
        tests/two_view_reconstruction_test.cpp
        tests/unittest1.cpp)
