cmake_minimum_required(VERSION 3.3)
project(movie-creator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread")
find_package( OpenCV REQUIRED core imgproc highgui photo )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include(../dlib/cmake)
#Internal files
GENERATE_PROJECT_STRUCTURE_VARIABLE()
add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_FILES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} face-morpher )
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_INCLUDES_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_HEADERS_DIR})
add_dependencies (${PROJECT_NAME} face-morpher)