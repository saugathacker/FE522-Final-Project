cmake_minimum_required(VERSION 3.20)
project(finalProject)

include(FetchContent)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

# Fetch Boost Math
FetchContent_Declare(
  boost_math
  GIT_REPOSITORY https://github.com/boostorg/math.git
  GIT_TAG boost-1.83.0  # Use the version you prefer
)
FetchContent_MakeAvailable(boost_math)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(libmatrix INTERFACE)
target_include_directories(libmatrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/matrix)
add_library(libfilehandler INTERFACE)
target_include_directories(libfilehandler INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/filehandler)
add_library(linearRegression INTERFACE)
target_include_directories(linearRegression INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/linearRegression ${boost_math_SOURCE_DIR}/include)
add_subdirectory(test)

add_executable(main main.cpp)
target_link_libraries(main libmatrix libfilehandler linearRegression)

# Add custom command to copy the data directory to the build directory
add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data
)