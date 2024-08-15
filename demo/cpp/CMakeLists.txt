cmake_minimum_required(VERSION 3.1...3.29)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  ModernCMakeExample
  VERSION 1.0
  LANGUAGES CXX)

file(
    GLOB_RECURSE
    MY_SRC
    "src/*.cpp"
    "src/*.h"
)

# print all files
message(STATUS "MY_SRC: ${MY_SRC}")

add_library(MyLibExample ${MY_SRC})

# Link each target with other targets or add options, etc.

# Adding something we can run - Output name matches target name
add_executable(Main main.cpp)

# Link the executable to the library
target_link_libraries(Main PRIVATE MyLibExample)

# add debug info
set(CMAKE_BUILD_TYPE Debug)

