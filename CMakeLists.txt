cmake_minimum_required(VERSION 3.19)
project(CourseWork)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CourseWork main.cpp)

target_link_libraries(CourseWork ${OpenCV_LIBS})
target_link_libraries(CourseWork nlohmann_json)