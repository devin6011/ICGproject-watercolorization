cmake_minimum_required(VERSION 2.8)
project(Clustering)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB Clustering_SRC
	"*.hpp"
	"*.cpp"
)
add_executable(Clustering ${Clustering_SRC})
target_link_libraries(Clustering ${OpenCV_LIBS} stdc++fs)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
