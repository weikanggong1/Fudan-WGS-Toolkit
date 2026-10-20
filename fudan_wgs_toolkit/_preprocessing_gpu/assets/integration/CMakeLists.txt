cmake_minimum_required(VERSION 3.19)
# CMake 3.21 adds cudadevrt even for Shared. Our kernels use neither RDC nor
# dynamic parallelism, and this Conda toolkit intentionally ships shared cudart.
# Override the generated link rule, including the compiler try_compile test,
# without altering installed CMake modules or adding a second CUDA runtime.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fnit_cuda_shared_rules.cmake"
  "set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED cudart)\n")
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA
  "${CMAKE_CURRENT_BINARY_DIR}/fnit_cuda_shared_rules.cmake")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architecture for the authorized A100")
endif()
project(fnit_pairhmm_integration LANGUAGES CXX CUDA)
if(NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Explicit Conda build environment is required")
endif()
find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../native" "${CMAKE_BINARY_DIR}/pairhmm_native")
add_library(fnit_pairhmm_jni SHARED pairhmm_jni.cpp)
target_compile_features(fnit_pairhmm_jni PRIVATE cxx_std_17)
target_include_directories(fnit_pairhmm_jni PRIVATE ${JNI_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/../native")
target_link_libraries(fnit_pairhmm_jni PRIVATE pairhmm_cuda CUDA::cudart)
set_target_properties(fnit_pairhmm_jni PROPERTIES BUILD_RPATH "$ORIGIN/pairhmm_native;$ENV{CONDA_PREFIX}/lib" INSTALL_RPATH "$ORIGIN")
install(TARGETS fnit_pairhmm_jni pairhmm_cuda LIBRARY DESTINATION lib)
