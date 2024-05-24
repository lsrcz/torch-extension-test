cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(torch-extension-test LANGUAGES C CXX CUDA VERSION 1.0)

set(CUDA_ARCHITECTURES 90)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
message(STATUS "TORCH_INTERFACE_LIB: ${TORCH_INTERFACE_LIB}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES "${TORCH_INTERFACE_LIB}")

add_library(lltm_ext MODULE
    lltm_cpp/lltm.cpp lltm_cpp/lltm_cuda.cpp lltm_cpp/lltm_cuda_kernel.cu lltm_cpp/lltm.h lltm_cpp/lltm_cpu.cpp 
)
python_extension_module(lltm_ext)
target_link_libraries(lltm_ext ${TORCH_LIBRARIES})
target_include_directories(lltm_ext PRIVATE ${TORCH_INCLUDE_DIRS})

