# Inpaint

**Inpaint** is a C++ library providing implementations of image inpainting methods. Image inpainting is the process of recovering or restoring
image regions in a way that is non-detectable for an observer who does not know the original image. 

**Inpaint** focuses on the task of object removal and is therefore optimized to work with larger areas of reconstruction.

## Building from source
To build **Inpaint** from source you need the following prerequisites
 - [CMake](www.cmake.org) - for generating cross plattform build files
 - [OpenCV](www.opencv.org) - for image processing related functions
 
Although **Inpaint** should build accross multiple platforms and architectures, tests are carried out on these systems
 - Windows 7/8 MSVC10 x86

If the build should fail for a specific platform, don't hestitate to create an issue. I'm also happy to accept any pull requests.
