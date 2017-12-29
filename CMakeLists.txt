cmake_minimum_required(VERSION 2.8)
project( tracker )
OPTION (HAVE_CUDA "Use Cuda" OFF)
OPTION (HAVE_INTEL "Use INTEL" OFF)

#if(HAVE_INTEL)
#	SET(MKLROOT "/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl")
#	SET (CMAKE_CXX_COMPILER             "icpc")
#	SET (CMAKE_CXX_FLAGS                "- -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl")
#else()
	SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
	SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
	#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
	SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
	SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
	SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
	SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
#endif()

find_package( OpenCV REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/detector/" )
include_directories( "src/likelihood/" )
include_directories( "src/utils/" )
include_directories( "src/DPP/" )
include_directories( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )

add_subdirectory("src/libs/piotr_fhog")

set(DETECTOR_GPU_SOURCES
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/multivariate_gaussian.cpp 
	src/detector/GPU_LR_hog_detector.cpp
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
	src/DPP/nms.hpp
    )

set(LR_CPU_SOURCES
	src/utils/c_utils.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/CPU_logistic_regression.cpp
	src/likelihood/multivariate_gaussian.cpp 
	)

if(HAVE_CUDA)
	find_package(CUDA QUIET REQUIRED)
	include_directories( "/usr/local/cuda/include/" )
endif()

if(HAVE_INTEL)
	include_directories( {MKLROOT}/include )
endif()

if(HAVE_CUDA)
	add_executable( detector_GPU src/test_detector.cpp ${DETECTOR_GPU_SOURCES})
 	target_link_libraries( detector_GPU ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} fhog)
else()
	#if(HAVE_INTEL)
	#	add_executable( detector_CPU src/test_detector_inria.cpp ${DETECTOR_CPU_SOURCES}  ${MKLROOT}/include)
	#	target_link_libraries( detector_CPU ${OpenCV_LIBS} fhog -DMKL_ILP64)
	#else()
		add_executable( detector_inria src/test_detector_inria.cpp ${DETECTOR_CPU_SOURCES}  )
		target_link_libraries( detector_inria ${OpenCV_LIBS} fhog)

		add_executable( detector_mars src/test_detector_mars.cpp ${DETECTOR_CPU_SOURCES}  )
		target_link_libraries( detector_mars ${OpenCV_LIBS} fhog)
	#endif()
endif()

