cmake_minimum_required(VERSION 3.18)
project(OpenFrameFlow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(RIFE_ROOT "" CACHE PATH "Checkout/build root of rife-ncnn-vulkan")
set(NCNN_ROOT "$ENV{HOME}/.local" CACHE PATH "Installation root of ncnn")
if(NOT RIFE_ROOT)
    message(FATAL_ERROR "Set -DRIFE_ROOT=/path/to/rife-ncnn-vulkan")
endif()

find_path(RIFE_INCLUDE_DIR rife.h PATHS "${RIFE_ROOT}/src" NO_DEFAULT_PATH REQUIRED)
find_path(NCNN_INCLUDE_DIR net.h
    PATHS "${RIFE_ROOT}/src/ncnn/src" "${RIFE_ROOT}/src/ncnn/build/src"
    NO_DEFAULT_PATH REQUIRED)
set(NCNN_GENERATED_INCLUDE_DIR "${RIFE_ROOT}/src/ncnn/build/src")
find_library(RIFE_LIBRARY NAMES rife-ncnn-vulkan
    PATHS "${RIFE_ROOT}/build" "${RIFE_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(NCNN_LIBRARY NAMES ncnn
    PATHS "${NCNN_ROOT}/lib" "${RIFE_ROOT}/src/ncnn/build/src"
    NO_DEFAULT_PATH REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

add_executable(open-frameflow src/server.cpp)
target_include_directories(open-frameflow PRIVATE
    "${RIFE_INCLUDE_DIR}" "${NCNN_INCLUDE_DIR}" "${NCNN_GENERATED_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(open-frameflow PRIVATE
    "${RIFE_LIBRARY}" "${NCNN_LIBRARY}" Vulkan::Vulkan Threads::Threads
    OpenMP::OpenMP_CXX ${OpenCV_LIBS})
target_compile_options(open-frameflow PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(open-frameflow PROPERTIES BUILD_RPATH "${RIFE_ROOT}/build")

install(TARGETS open-frameflow RUNTIME DESTINATION bin)
