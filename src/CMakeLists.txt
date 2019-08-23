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
	shomagick 
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
  target_include_directories(shomagick PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries (shomagick PRIVATE ${CUDA_LIBRARIES})
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
  target_include_directories(shomagick ${SUITESPARSE_INCLUDE_DIRS})
endif()



target_include_directories(shomagick
	PUBLIC
	 	$<INSTALL_INTERFACE:include>  
		${EIGEN_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${OPENGV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

	PRIVATE
		${EXIV2_INCLUDE_DIRS}
		${CERES_INCLUDE_DIR}
		${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(
	shomagick
	PUBLIC
		${Boost_LIBRARIES}
		${OpenCV_LIBS}
		${OPENGV_LIBRARIES}

	PRIVATE
		${CERES_LIBRARIES}
		${LAPACK_LIBRARIES}
		${SUITESPARSE_LIBRARIES}
		${EXIV2_LIBRARIES}
		vl
)

set(INSTALL_CONFIGDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/Shomagick)


install(TARGETS shomagick vl EXPORT shomagick-targets DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(shomagick PROPERTIES EXPORT_NAME Shomagick)

install(EXPORT shomagick-targets
  FILE
    ShomagickTargets.cmake
  NAMESPACE
    Shomagick::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ShomagickConfigVersion.cmake
    VERSION ${SHOMAGICK_MAJOR_VERSION}.${SHOMAGICK_MINOR_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/ShomagickConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ShomagickConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, configversion and custom find modules
install(FILES
    #${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ShomagickConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ShomagickConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

##############################################
## Exporting from the build tree
#configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake
    #${CMAKE_CURRENT_BINARY_DIR}/FindRapidJSON.cmake
    #COPYONLY)

export(EXPORT shomagick-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/ShomagickTargets.cmake NAMESPACE Shomagick::)

#Register package in user's package registry
export(PACKAGE Shomagick)