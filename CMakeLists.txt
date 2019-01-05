cmake_minimum_required(VERSION 2.8)
project(Watercolorization)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB Watercolorization_SRC
	"*.hpp"
	"*.cpp"
)
add_executable(Watercolorization ${Watercolorization_SRC})
target_link_libraries(Watercolorization ${OpenCV_LIBS})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
