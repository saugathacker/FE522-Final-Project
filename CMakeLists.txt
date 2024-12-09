cmake_minimum_required(VERSION 3.20)
project(finalProject)

include(FetchContent)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(libmatrix INTERFACE)
target_include_directories(libmatrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(test)

add_executable(main main.cpp)
target_link_libraries(main libmatrix)