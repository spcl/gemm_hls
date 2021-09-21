# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Copyright: This software is copyrighted under the BSD 3-Clause License.

cmake_minimum_required(VERSION 3.0)
project(sdaccel_mm)

# Target options 
set(MM_PLATFORM "xilinx_u250_gen3x16_xdma_3_1_202020_1" CACHE STRING "Platform string for Vitis.")
set(MM_TARGET_CLOCK "" CACHE STRING "Target clock for kernel (uses default is left empty).")
set(MM_ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information.")
set(MM_ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design.")
set(MM_VITIS_FLAGS "" CACHE STRING "Extra flags for Vitis.")
set(MM_POWER_METER OFF CACHE BOOL "Enable Corsair power meter for measuring power consumption with Corsair RMi power supplies.")
set(MM_ENABLE_BLAS ON CACHE BOOL "Use BLAS library for validation if available.")

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(MM_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(MM_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(MM_DYNAMIC_SIZES ON CACHE BOOL "Use dynamic matrix dimension sizes.")
set(MM_SIZE_N 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_K 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 512 CACHE STRING "Size of matrix dimension.")
set(MM_MEMORY_TILE_SIZE_N 256 CACHE STRING "Tile size of outer memory tile in N.")
set(MM_MEMORY_TILE_SIZE_M 256 CACHE STRING "Tile size of outer memory tile in M.")
set(MM_PARALLELISM_N 32 CACHE STRING "Number of parallel compute in N.")
set(MM_PARALLELISM_M 8 CACHE STRING "Number of parallel compute in M.")
# set(MM_GRANULARITY_N 1 CACHE STRING "Granularity of processing elements in N.")
set(MM_TRANSPOSED_A OFF CACHE BOOL "Assume the input matrix A is transposed.")
set(MM_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(MM_TWO_DIMMS OFF CACHE BOOL "Use two DDR DIMMs instead of one")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULT_RESOURCE OFF CACHE STRING "")  

# Internal
set(MM_KERNEL_NAME MatrixMultiplicationKernel)
include(CheckTypeSize)
check_type_size(${MM_DATA_TYPE} MM_DATA_WIDTH_${MM_DATA_TYPE})
if(NOT MM_DATA_WIDTH_${MM_DATA_TYPE}) # Non-primitive data type
  if(MM_DATA_TYPE STREQUAL "half")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 2)
  elseif(MM_DATA_TYPE STREQUAL "uint8_t")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 1)
  else()
    message(FATAL_ERROR "Could not get size of data type ${MM_DATA_TYPE}.") 
  endif()
endif()
math(EXPR MM_KERNEL_WIDTH_N "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_N}")
math(EXPR MM_KERNEL_WIDTH_M "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_M}")

# Validation checks
math(EXPR MM_INNER_TILES "(${MM_MEMORY_TILE_SIZE_N} / ${MM_PARALLELISM_N}) * (${MM_MEMORY_TILE_SIZE_M} / ${MM_PARALLELISM_M})")
if(NOT MM_TRANSPOSED_A AND MM_MEMORY_TILE_SIZE_N GREATER MM_INNER_TILES)
  message(WARNING "In-memory transposition for A cannot keep up with the instantiated number of compute units. The number of inner tiles (currently ${MM_INNER_TILES}) must be greater than or equal to the outer tile size in N (currently ${MM_MEMORY_TILE_SIZE_N}).")
endif()
math(EXPR MM_MEMORY_WIDTH_M "${MM_MEMORY_BUS_WIDTH_M} / ${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
math(EXPR MM_REM "${MM_MEMORY_WIDTH_M} % ${MM_PARALLELISM_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Bus width in M (${MM_MEMORY_WIDTH_M}) must be a multiple of the parallelism in M (${MM_PARALLELISM_M}).")
endif()
math(EXPR MM_REM "${MM_MEMORY_TILE_SIZE_M} % ${MM_MEMORY_WIDTH_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Outer memory tile size must be divisible by element width of memory bus.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)

# Check if BLAS is available
if(MM_ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMM_HAS_BLAS")
  else()
    set(BLAS_LIBRARIES)
  endif()
else()
  set(BLAS_LIBRARIES)
endif()

# Include power meter
if(MM_POWER_METER)
  add_subdirectory(powermeter)
  include_directories(SYSTEM powermeter/include powermeter/OpenCorsairLink/include ${LIBUSB_INCLUDE_DIR})
  add_definitions("-DMM_POWER_METER")
endif()

include_directories(include ${CMAKE_BINARY_DIR} SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
if(MM_DYNAMIC_SIZES)
  add_definitions("-DMM_DYNAMIC_SIZES")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_DYNAMIC_SIZES")
endif()
if(MM_TRANSPOSED_A)
  add_definitions("-DMM_TRANSPOSED_A")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_TRANSPOSED_A")
endif()
add_definitions("-DHLSLIB_STREAM_TIMEOUT=16")
if(((${Vitis_MAJOR_VERSION} LESS 2018) AND (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
  add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
  add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  add_definitions("-DMM_HALF_PRECISION")
endif()

# Query default clock of platform
if(NOT "${MM_TARGET_CLOCK}")
  if(NOT "${MM_PLATFORM_INTERNAL}" STREQUAL "${MM_PLATFORM}")
    message(STATUS "Querying default clock for ${MM_PLATFORM}.")
    execute_process(COMMAND ${Vitis_PLATFORMINFO} --platform ${MM_PLATFORM} -jhardwarePlatform.systemClocks
                    OUTPUT_VARIABLE SYSTEM_CLOCKS
                    RESULT_VARIABLE RET)
    if(RET EQUAL 0)
      string(FIND "${SYSTEM_CLOCKS}" "\"default\": \"true\"" LOC)
      string(SUBSTRING "${SYSTEM_CLOCKS}" ${LOC} -1 SYSTEM_CLOCKS)
      string(REGEX MATCH "\"frequency\": \"[0-9\\.]+\"" SYSTEM_CLOCKS "${SYSTEM_CLOCKS}")
      string(REGEX REPLACE "\"frequency\": \"([0-9\\.]+)\"" "\\1" CLOCK "${SYSTEM_CLOCKS}")
      set(MM_CLOCK_INTERNAL "${CLOCK}" CACHE INTERNAL "")
      set(MM_PLATFORM_INTERNAL "${MM_PLATFORM}" CACHE INTERNAL "")
    else()
      message(WARNING "Failed to query default frequency for platform ${MM_PLATFORM}. Assuming 300 MHz.")
      set(MM_CLOCK_INTERNAL "300" CACHE INTERNAL "")
    endif()
  endif()
endif()

# Hardware configuration header for HLS
configure_file(include/Config.h.in Config.h)

# C++ source code 
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Top.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

# Executables 
add_executable(PrintSpecifications src/PrintSpecifications.cpp)

# Software test
add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} ${Vitis_LIBRARIES} mmkernel)
if(MM_DATA_TYPE STREQUAL "half")
  target_link_libraries(TestSimulation ${Vitis_FLOATING_POINT_LIBRARY})
endif()
if(MM_DYNAMIC_SIZES)
  math(EXPR TEST_SIZE_N "2 * ${MM_MEMORY_TILE_SIZE_N} + 1")
  math(EXPR TEST_SIZE_K "2 * ${MM_PARALLELISM_N} * ${MM_PARALLELISM_M} + ${MM_MEMORY_BUS_WIDTH_K} / ${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
  math(EXPR TEST_SIZE_M "2 * ${MM_MEMORY_TILE_SIZE_M} + ${MM_MEMORY_WIDTH_M}")
  add_test(TestSimulation TestSimulation ${TEST_SIZE_N} ${TEST_SIZE_K} ${TEST_SIZE_M})
else()
  add_test(TestSimulation TestSimulation)
endif()
enable_testing()

# Synthesis flags
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -O3 -DMM_SYNTHESIS")
if(MM_DATA_TYPE STREQUAL "half")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_HALF_PRECISION")
endif()

if(MM_ADD_RESOURCE)
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
endif() 
if(MM_MULT_RESOURCE) 
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_MULT_RESOURCE=${MM_MULT_RESOURCE}") 
endif() 

if(MM_TWO_DIMMS)
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[0] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
else()
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[1] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
endif()

# Hardware kernel
add_vitis_kernel(MatrixMultiplication
                 KERNEL "MatrixMultiplicationKernel"
                 FILES ${MM_KERNEL_SRC}
                 HLS_FLAGS ${MM_SYNTHESIS_FLAGS}
                 INCLUDE_DIRS include
                              hlslib/include
                              ${CMAKE_BINARY_DIR}
                 PORT_MAPPING ${MM_PORT_MAPPING} 
                 DEPENDS include/Compute.h
                         include/MatrixMultiplication.h
                         include/Memory.h
                         ${CMAKE_BINARY_DIR}/Config.h)
add_vitis_program(MatrixMultiplication ${MM_PLATFORM}
                  CLOCK ${MM_TARGET_CLOCK}
                  BUILD_FLAGS ${MM_VITIS_FLAGS}
                  PROFILING ${MM_ENABLE_PROFILING}
                  DEBUGGING ${MM_ENABLE_DEBUGGING})

# Host code to launch kernel
add_executable(RunHardware.exe host/RunHardware.cpp)
target_link_libraries(RunHardware.exe ${Vitis_LIBRARIES} ${BLAS_LIBRARIES} mmkernel)
if(MM_TWO_DIMMS)
  target_compile_definitions(RunHardware.exe PRIVATE MM_TWO_DIMMS)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  target_link_libraries(RunHardware.exe ${Vitis_FLOATING_POINT_LIBRARY})
endif()
if(MM_POWER_METER)
  target_link_libraries(RunHardware.exe powermeter)
endif()
