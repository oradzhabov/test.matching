cmake_minimum_required(VERSION 2.8)
project(test.matching)

find_package( OpenCV REQUIRED COMPONENTS core features2d imgproc imgcodecs highgui ml photo flann calib3d)

MESSAGE("OpenCV information:") 
MESSAGE(STATUS "  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
MESSAGE(STATUS "  OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}") 
MESSAGE(STATUS "  OpenCV_LIBRARY_DIRS: ${OpenCV_LIBRARY_DIRS}") 
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenGL REQUIRED )


add_subdirectory(projects/rd.1)
add_subdirectory(projects/rd.2)

set_target_properties( rd.1 rd.2
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
