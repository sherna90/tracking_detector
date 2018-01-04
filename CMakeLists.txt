cmake_minimum_required(VERSION 2.8)
project( tracker )
OPTION (HAVE_CUDA "Use Cuda" OFF)
OPTION (HAVE_INTEL "Use INTEL" OFF)

SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

find_package( OpenCV REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/detector/" )
include_directories( "src/likelihood/" )
include_directories( "src/utils/" )
include_directories( "src/DPP/" )
include_directories( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )

add_subdirectory("src/libs/piotr_fhog")



set(DETECTOR_CPU_SOURCES
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/CPU_logistic_regression.cpp
	src/likelihood/multivariate_gaussian.cpp	
	src/detector/hog_detector.cpp   
	src/detector/CPU_LR_hog_detector.cpp  
	src/DPP/dpp.cpp
	src/DPP/nms.hpp
    )

#add_executable( detector_inria src/test_detector_inria.cpp ${DETECTOR_CPU_SOURCES}  )
#target_link_libraries( detector_inria ${OpenCV_LIBS} fhog)

add_executable( detector_mars src/test_detector_mars.cpp ${DETECTOR_CPU_SOURCES}  )
target_link_libraries( detector_mars ${OpenCV_LIBS} fhog)

