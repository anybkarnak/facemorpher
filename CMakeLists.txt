cmake_minimum_required(VERSION 3.3)
project(face_morpher)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

MESSAGE( STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY:         " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
MESSAGE( STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY:         " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )

#Include utils
include(CMakeFunc.txt)
include(dlib/cmake)
add_subdirectory(face-landmark-detector)
add_subdirectory(face-morpher)
add_subdirectory(movie-creator)
add_subdirectory(dlib)
set(SOURCE_FILES main.cpp)
add_executable(face_morpher ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} face-morpher face-landmark-detector movie-creator)


add_dependencies (${PROJECT_NAME} movie-creator)