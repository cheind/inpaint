cmake_minimum_required(VERSION 2.8)

project(inpainting)

if (WIN32)
	#set(OpenCV_STATIC true)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} "inc")


# Library

add_library(inpaint 
	inc/inpaint/stats.h
	inc/inpaint/patch.h
	inc/inpaint/gradient.h
    inc/inpaint/integral.h
    inc/inpaint/timer.h
	inc/inpaint/criminisi_inpainter.h
	inc/inpaint/template_match_candidates.h
	inc/inpaint/patch_match.h
	src/criminisi_inpainter.cpp
	src/template_match_candidates.cpp
	src/patch_match.cpp
)
	
target_link_libraries(inpaint ${OpenCV_LIBRARIES})
	
# Samples

add_executable(inpaint_image_criminisi examples/inpaint_image_criminisi.cpp)
target_link_libraries(inpaint_image_criminisi inpaint ${OpenCV_LIBRARIES})

add_executable(patch_match examples/patch_match.cpp)
target_link_libraries(patch_match inpaint ${OpenCV_LIBRARIES})

# Tests

include_directories("tests")
add_executable(inpaint_tests 
	tests/catch.hpp
	tests/random_image.h
	tests/gradient.cpp
	tests/patch.cpp
    tests/integral.cpp
	tests/criminisi_inpainter.cpp
    tests/template_match_candidates.cpp
)
target_link_libraries (inpaint_tests inpaint ${OpenCV_LIBRARIES})

# Benchmarks

include_directories("benchmarks")
add_executable(inpaint_benchmarks 
	benchmarks/catch.hpp
	benchmarks/patch.cpp	
)
target_link_libraries (inpaint_benchmarks inpaint ${OpenCV_LIBRARIES})