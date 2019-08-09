#Open SFM vlfeat
include_directories(vlfeat)
file(GLOB VLFEAT_SRCS vlfeat/vl/*.c vlfeat/vl/*.h)
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES
    "(x86)|(X86)|(x86_64)|(X86_64)|(amd64)|(AMD64)")
    add_definitions(-DVL_DISABLE_SSE2)
endif ()
add_definitions(-DVL_DISABLE_AVX)

add_library(vl ${VLFEAT_SRCS})

add_library(
	Shomagick 
	RobustMatcher.cpp 
	HahogFeatureDetector.cpp 
	allclose.cpp camera.cpp 
	flightsession.cpp  
	image.cpp 
	kdtree.cpp 
	multiview.cpp 
	reconstruction.cpp 
	reconstructor.cpp  	
	shomatcher.cpp 
	transformations.cpp  
	utilities.cpp
)

find_package(Boost 1.5 COMPONENTS filesystem serialization REQUIRED)

find_package(Eigen REQUIRED)

find_package(exiv2 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package (CUDA)
if (CUDA_FOUND)
  target_include_directories(Shomagick PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries (Shomagick PRIVATE ${CUDA_LIBRARIES})
endif (CUDA_FOUND)

find_package(
	OpenCV 
	REQUIRED 
	COMPONENTS
	core 
	imgproc 
	calib3d 
	sfm 
	viz 
	features2d 
	xfeatures2d 
	OPTIONAL_COMPONENTS
	cudafeatures2d 
	cudaimgproc 
)

find_package(Ceres REQUIRED)

find_package(OpenGV REQUIRED)

find_package(SuiteSparse)
if (SUITESPARSE_FOUND)
  target_include_directories(Shomagick ${SUITESPARSE_INCLUDE_DIRS})
endif()



target_include_directories(Shomagick
	PUBLIC
		${EIGEN_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/include

	PRIVATE
		${EXIV2_INCLUDE_DIRS}
		${OPENGV_INCLUDE_DIRS}
		${CERES_INCLUDE_DIR}
		${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(
	Shomagick
	PUBLIC
		${Boost_LIBRARIES}
		${OpenCV_LIBS}

	PRIVATE
		${CERES_LIBRARIES}
		${LAPACK_LIBRARIES}
		${SUITESPARSE_LIBRARIES}
		${EXIV2_LIBRARIES}
		${OPENGV_LIBRARIES}
		vl
)
