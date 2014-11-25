cmake_minimum_required(VERSION 2.8)

project(inpainting)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} "inc")

# Library

add_library(inpaint 
	inc/inpaint/criminisi_inpainter.h
	src/criminisi_inpainter.cpp)
	
target_link_libraries (inpaint ${OpenCV_LIBRARIES})
	
# Samples

add_executable(inpaint_image_criminisi examples/inpaint_image_criminisi.cpp)
target_link_libraries (inpaint_image_criminisi inpaint ${OpenCV_LIBRARIES})