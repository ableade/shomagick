cmake_minimum_required (VERSION 2.6)
project (Shomagick)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC  -fvisibility=hidden -fvisibility-inlines-hidden")

set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${SOURCE_DIR}/cmake/Modules")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BUNDLE_SOURCE_DIR "${SOURCE_DIR}/bundle")

set(CUDA_SOURCES
  ${SOURCE_DIR}/cudasift/cudaImage.cu  
  ${SOURCE_DIR}/cudasift/cudaImage.h  
  ${SOURCE_DIR}/cudasift/cudaSiftH.cu 
  ${SOURCE_DIR}/cudasift/cudaSiftH.h  
  ${SOURCE_DIR}/cudasift/matching.cu  
  ${SOURCE_DIR}/cudasift/cudaSiftD.h  
  ${SOURCE_DIR}/cudasift/cudaSift.h  
  ${SOURCE_DIR}/cudasift/cudautils.h
)  

include_directories( ${INCLUDE_DIR} )


set(Shomagick_VERSION_MAJOR 1)
set(Shomagick_VERSION_MINOR 0)

configure_file (
  "${SOURCE_DIR}/ShomagickConfig.h.in"
  "${PROJECT_BINARY_DIR}/ShomagickConfig.h"
)

include(GNUInstallDirs)
include_directories("${PROJECT_BINARY_DIR}")

#Open SFM vlfeat
include_directories(${SOURCE_DIR}/vlfeat)
file(GLOB VLFEAT_SRCS ${SOURCE_DIR}/vlfeat/vl/*.c ${SOURCE_DIR}/vlfeat/vl/*.h)

add_library(vl ${VLFEAT_SRCS})

add_library(
	shomagick 
	${SOURCE_DIR}/RobustMatcher.cpp 
	${SOURCE_DIR}/HahogFeatureDetector.cpp 
	${SOURCE_DIR}/allclose.cpp 
	${SOURCE_DIR}/camera.cpp 
	${SOURCE_DIR}/flightsession.cpp  
	${SOURCE_DIR}/image.cpp 
	${SOURCE_DIR}/kdtree.cpp 
	${SOURCE_DIR}/multiview.cpp 
	${SOURCE_DIR}/reconstruction.cpp 
	${SOURCE_DIR}/reconstructor.cpp  	
	${SOURCE_DIR}/shomatcher.cpp
	${SOURCE_DIR}/shot.cpp  
	${SOURCE_DIR}/transformations.cpp  
	${SOURCE_DIR}/shotracking.cpp  
	${SOURCE_DIR}/utilities.cpp
	${SOURCE_DIR}/cudaSurfFeatureDetector.cpp
	${SOURCE_DIR}/cudaSiftFeatureDetector.cpp
)

target_compile_features(shomagick PRIVATE cxx_std_17)

if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES
    "(x86)|(X86)|(x86_64)|(X86_64)|(amd64)|(AMD64)")
    target_compile_definitions(vl PRIVATE VL_DISABLE_SSE2)
endif ()
target_compile_definitions(vl PRIVATE VL_DISABLE_AVX)

IF(NOT MSVC)
	message("Not on MSVC")
ENDIF()

IF(MSVC)
  target_compile_options(vl PRIVATE /wd4514 /wd4267 /bigobj)
  target_compile_definitions(vl PRIVATE _USE_MATH_DEFINES)
ELSE()
  IF (CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64)|(ARM64)|(aarch64)|(AARCH64)")
    target_compile_options (vl PRIVATE -march=armv8-a)
  ELSEIF (CMAKE_SYSTEM_PROCESSOR MATCHES 
          "(arm)|(ARM)|(armhf)|(ARMHF)|(armel)|(ARMEL)")
	target_compile_options(vl PRIVATE -march=armv7-a)
  ELSE ()
    target_compile_options (vl PRIVATE -march=native) #TODO use correct c++11 def once everybody has moved to gcc 4.7 # for now I even removed std=gnu++0x
  ENDIF()
  target_compile_options (vl
  	PRIVATE
		-O3
		-Wall
		-Wextra
		#-Werror
		-Wwrite-strings
		-Wno-unused-parameter
		-fno-strict-aliasing
  )
ENDIF()

find_package(Boost 1.5 COMPONENTS filesystem serialization REQUIRED)

find_package(Eigen REQUIRED)

find_package(exiv2 REQUIRED)

find_library(exiv_lib exiv2)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#set(CUDA_SEPARABLE_COMPILATION ON)
find_package (CUDA)
if (CUDA_FOUND)
	enable_language(CUDA)
	message("CUDA FOUND.")
	if(UNIX)
		if (APPLE)
			set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
			list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  		else()
    		#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    		list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  		endif()
		message("Compiling sift cuda")
		#Only compile the sift cuda library if on a non-windows platform
		cuda_add_library(cudasift ${CUDA_SOURCES})
		target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
		#target_link_libraries(cudasift PUBLIC ${CUDA_LIBRARIES})
		#target_link_libraries(cudasift PUBLIC ${CUDA_cudadevrt_LIBRARY})
		target_include_directories(cudasift PUBLIC ${CUDA_INCLUDE_DIRS})
		target_compile_definitions(cudasift PUBLIC USE_GPU)
		target_link_libraries(shomagick PUBLIC cudasift)
	endif()
	target_include_directories(shomagick PUBLIC ${CUDA_INCLUDE_DIRS})
  	target_link_libraries (shomagick PUBLIC ${CUDA_LIBRARIES})
endif (CUDA_FOUND)

find_package(
	OpenCV 
	REQUIRED 
	COMPONENTS
		core 
		imgproc
		highgui 
		calib3d 
		sfm 
		viz 
		imgcodecs
		features2d 
		xfeatures2d 
	OPTIONAL_COMPONENTS
		cudafeatures2d 
		cudaimgproc 
)

find_package(Ceres REQUIRED)

find_package(opengv REQUIRED)

find_package(Glog REQUIRED)

find_package(OpenMP REQUIRED)

find_package(Gflags REQUIRED)

find_package(SuiteSparse)
if (SUITESPARSE_FOUND)
  target_include_directories(shomagick PRIVATE ${SUITESPARSE_INCLUDE_DIRS})
  target_link_libraries(shomagick PRIVATE ${SUITESPARSE_LIBRARIES})
endif()


target_include_directories(shomagick
	PUBLIC
	 	$<INSTALL_INTERFACE:include/>  
		${EIGEN_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${OPENGV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		${GFLAGS_INCLUDE_DIR}
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>	
		${EXIV2_INCLUDE_DIRS}
		${CERES_INCLUDE_DIR}
		${GLOG_INCLUDE_DIRS}

	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_link_libraries(
	shomagick
	PUBLIC
		${Boost_LIBRARIES}
		${OpenCV_LIBS}
		opengv
		${GFLAGS_LIBRARY}

	
		${CERES_LIBRARIES}
		${LAPACK_LIBRARIES}
		${SUITESPARSE_LIBRARIES}
		exiv2
		${GLOG_LIBRARIES}
		${OpenMP_CXX_LIBRARIES}
		vl
)

get_target_property(CONF_INCLUDE_DIRS shomagick INTERFACE_INCLUDE_DIRECTORIES)
set(INSTALL_CONFIGDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/Shomagick)

if (CUDA_FOUND AND UNIX)
	target_link_libraries(shomagick PUBLIC cudasift)
	set(LIBS_TARGET_SET shomagick vl cudasift)
else()
	set(LIBS_TARGET_SET shomagick vl)
endif()
	
install(TARGETS ${LIBS_TARGET_SET} EXPORT shomagick-targets DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
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

configure_package_config_file(${SOURCE_DIR}/cmake/ShomagickConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ShomagickConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, configversion and custom find modules
install(FILES
	${SOURCE_DIR}/cmake/Modules/FindCeres.cmake
	${SOURCE_DIR}/cmake/Modules/FindEigen.cmake
	${SOURCE_DIR}/cmake/Modules/FindGflags.cmake
	${SOURCE_DIR}/cmake/Modules/FindGlog.cmake
	${SOURCE_DIR}/cmake/Modules/FindSuiteSparse.cmake
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

install(DIRECTORY include/  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Shomagick)
