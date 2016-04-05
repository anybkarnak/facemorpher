cmake_minimum_required(VERSION 3.3)
project(face-landmark-detector)
include(../dlib/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
#Internal files
GENERATE_PROJECT_STRUCTURE_VARIABLE()
install_files(${CMAKE_RUNTIME_OUTPUT_DIRECTORY} FILES "src/shape_predictor_68_face_landmarks.dat")
add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_FILES})
# Then we tell it to link with dlib.
TARGET_LINK_LIBRARIES(${PROJECT_NAME}  dlib)
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_INCLUDES_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_HEADERS_DIR})