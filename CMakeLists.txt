cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
OPTION (HAVE_CUDA "Use Cuda" OFF)
OPTION (MAKE_TESTS "Build Tests" OFF)

find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/detector/" )
include_directories( "src/likelihood/" )
include_directories( "src/utils/" )
include_directories( "src/DPP/" )
include_directories( "/usr/include/eigen3/" )
#include_directories( "/usr/local/include/eigen3/" )

set(DETECTOR_GPU_SOURCES
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/GPU_logistic_regression.cpp 
	src/likelihood/multivariate_gaussian.cpp 
	src/detector/hog_detector.cpp 
	src/detector/GPU_LR_hog_detector.cpp  
	src/detector/GPU_HMC_hog_detector.cpp 
	src/DPP/dpp.cpp
    )

set(DETECTOR_CPU_SOURCES
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/CPU_logistic_regression.cpp 
	src/likelihood/multivariate_gaussian.cpp 
	src/detector/hog_detector.cpp 
	src/detector/CPU_LR_hog_detector.cpp  
	src/DPP/dpp.cpp
    )

if(HAVE_CUDA)
	find_package(CUDA QUIET REQUIRED)
	include_directories( "/usr/local/cuda/include/" )
endif()

if(HAVE_CUDA)
	add_executable( detector_GPU src/test_detector.cpp ${DETECTOR_GPU_SOURCES})
 	target_link_libraries( detector_GPU ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
else()
	add_executable( detector_CPU src/test_detector_cpu.cpp ${DETECTOR_CPU_SOURCES}  )
	target_link_libraries( detector_CPU ${OpenCV_LIBS} )
endif()

if(MAKE_TESTS)
if(HAVE_CUDA)
	add_executable( test_lr_GPU src/likelihood/logistic_regression.cpp src/likelihood/Mask_GPU_logistic_regression.cpp src/likelihood/GPU_logistic_regression.cpp src/utils/c_utils.cpp  src/likelihood/multivariate_gaussian.cpp src/test_lr.cpp)
	target_link_libraries( test_lr_GPU ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

	add_executable( test_hmc_GPU src/likelihood/logistic_regression.cpp src/likelihood/GPU_logistic_regression.cpp src/likelihood/Mask_GPU_logistic_regression.cpp  src/likelihood/hmc.cpp src/likelihood/GPU_hmc.cpp src/likelihood/Mask_GPU_hmc.cpp src/utils/c_utils.cpp src/likelihood/multivariate_gaussian.cpp src/test_hmc.cpp)
	target_link_libraries( test_hmc_GPU ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
else()
	add_executable( test_lr_CPU src/likelihood/logistic_regression.cpp src/likelihood/Mask_CPU_logistic_regression.cpp src/likelihood/CPU_logistic_regression.cpp src/utils/c_utils.cpp  src/likelihood/multivariate_gaussian.cpp src/test_lr.cpp)
	target_link_libraries( test_lr_CPU ${OpenCV_LIBS} )

	#add_executable( test_hmc_CPU src/likelihood/logistic_regression.cpp src/likelihood/CPU_logistic_regression.cpp src/likelihood/Mask_CPU_logistic_regression.cpp src/likelihood/hmc.cpp  src/likelihood/CPU_hmc.cpp likelihood/Mask_CPU_hmc.cpp src/utils/c_utils.cpp src/likelihood/multivariate_gaussian.cpp src/test_hmc.cpp)
	#target_link_libraries( test_hmc_CPU ${OpenCV_LIBS} )
endif()
endif()
