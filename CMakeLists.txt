# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)

cmake_minimum_required(VERSION 3.0)
project(sdaccel_mm)

# Target options 
set(MM_PART_NAME "xcu280-fsvh2892-2L-e" CACHE STRING "Part name for HLS.")
set(MM_DSA_NAME "xilinx_u280_xdma_201920_1" CACHE STRING "DSA string for Vitis.")
set(MM_TARGET_CLOCK 200 CACHE STRING "Target clock for kernel (<=0 uses DSA default).")
set(MM_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(MM_TMP_DIR /tmp CACHE STRING "Temporary directory for SDx build files")
set(MM_ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information")
set(MM_ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design")
set(MM_QUICK_BUILD OFF CACHE BOOL "Enable quick build option for Vitis")
set(MM_VPP_FLAGS "" CACHE STRING "Extra flags for Vitis")
set(MM_POWER_METER OFF CACHE BOOL "Enable Corsair power meter for measuring power consumption with Corsair RMi power supplies.")

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(MM_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(MM_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(MM_DYNAMIC_SIZES OFF CACHE BOOL "Use dynamic matrix dimension sizes.")
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
set(MM_ENTRY_FUNCTION MatrixMultiplication)
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
math(EXPR MM_REM "${MM_SIZE_N} % ${MM_MEMORY_TILE_SIZE_N}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Size in N (${MM_SIZE_M}) must be divisible by memory tile size in N (${MM_MEMORY_TILE_SIZE_N}).")
endif()
math(EXPR MM_REM "${MM_SIZE_M} % ${MM_MEMORY_TILE_SIZE_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Size in M (${MM_SIZE_M}) must be divisible by memory tile size in M (${MM_MEMORY_TILE_SIZE_M}).")
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
find_package(BLAS)
if(BLAS_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMM_HAS_BLAS")
else()
  set(BLAS_LIBRARIES)
endif()

# Include power meter
if(MM_POWER_METER)
  add_subdirectory(powermeter)
  include_directories(SYSTEM powermeter/include powermeter/OpenCorsairLink/include ${LIBUSB_INCLUDE_DIR})
  add_definitions("-DMM_POWER_METER")
endif()

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${Vitis_INCLUDE_DIRS})
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
target_link_libraries(TestSimulation ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} ${Vitis_FLOATING_POINT_LIBRARY} ${Vitis_LIBRARIES} mmkernel)
add_test(TestSimulation TestSimulation)
enable_testing()

# Standalone synthesis target. Starting from Vitis 2019.2, this no longer works,
# as the tool insists on inlining the processing elements so they're not
# instantiated as dataflow functions.
if((${Vitis_MAJOR_VERSION} GREATER 2018 AND ${Vitis_MINOR_VERSION} LESS 2) OR ${Vitis_MAJOR_VERSION} LESS 2019) 
  set(MM_HLS_PROJECT_NAME "hls")
  string(REPLACE ";" " " MM_SRC_SYNTHESIS "${MM_KERNEL_SRC}")
  set(MM_ENTRY_FUNCTION "MatrixMultiplicationKernel")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -std=c++11 -O3 -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS")
  if(MM_ADD_RESOURCE)
    set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
  endif() 
  if(MM_MULT_RESOURCE) 
    set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_MULT_RESOURCE=${MM_MULT_RESOURCE}") 
  endif() 
  configure_file(scripts/Synthesis.tcl.in Synthesis.tcl)
  add_custom_target(synthesis COMMAND ${Vitis_HLS} -f Synthesis.tcl) 
  string(REPLACE " " ";" MM_SYNTHESIS_FLAGS ${MM_SYNTHESIS_FLAGS})
endif()

# Hardware kernel
string(REPLACE " " ";" MM_VPP_FLAGS "${MM_VPP_FLAGS}")
set(MM_VPP_COMMAND
  -s
  ${MM_VPP_FLAGS}
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  --kernel ${MM_KERNEL_NAME} 
  --platform ${MM_DSA_NAME}
  --xp prop:kernel.${MM_ENTRY_FUNCTION}.kernel_flags="${MM_SYNTHESIS_FLAGS}"
  --max_memory_ports all)
if(MM_ENABLE_PROFILING)
  set(MM_VPP_COMMAND ${MM_VPP_COMMAND}
    --profile_kernel "data:all:all:all"
    --profile_kernel "stall:all:all"
    --profile_kernel "exec:all:all")
endif()
if(MM_QUICK_BUILD)
  set(MM_VPP_COMMAND ${MM_VPP_COMMAND} -Oquick)
else()
  set(MM_VPP_COMMAND ${MM_VPP_COMMAND} -O3)
endif()
if(MM_ENABLE_DEBUGGING)
  set(MM_VPP_COMMAND ${MM_VPP_COMMAND} -g
    --dk chipscope:MatrixMultiplicationKernel_1:M_AXI_GMEM0
    --dk chipscope:MatrixMultiplicationKernel_1:M_AXI_GMEM1
    --dk chipscope:MatrixMultiplicationKernel_1:M_AXI_GMEM2)
endif()
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_VPP_COMMAND ${MM_VPP_COMMAND}
      --kernel_frequency ${MM_TARGET_CLOCK})
endif()
if(MM_TWO_DIMMS)
  if(NOT (MM_DSA_NAME MATCHES "dynamic"))
    set(MM_VPP_COMMAND ${MM_VPP_COMMAND}
      --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI
      --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI
      --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.M_AXI_GMEM2.core.OCL_REGION_0.M01_AXI)
  else()
    set(MM_VPP_COMMAND ${MM_VPP_COMMAND}
      --sp ${MM_KERNEL_NAME}_1.m_axi_gmem0:bank0
      --sp ${MM_KERNEL_NAME}_1.m_axi_gmem1:bank1
      --sp ${MM_KERNEL_NAME}_1.m_axi_gmem2:bank1)
  endif()
endif()

if(((${Vitis_MAJOR_VERSION} LESS 2018) AND (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
  add_custom_target(build_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} -t hw
    ${MM_VPP_COMMAND} ${MM_VPP_COMPILE_FLAGS} ${MM_VPP_LINK_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMultiplication_hw.xclbin) 
else()
  # Hardware
  add_custom_target(compile_hardware
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} -c -t hw
    ${MM_VPP_COMMAND} ${MM_VPP_LINK_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMultiplication_hw.xo) 
  add_custom_target(link_hardware
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} -l -t hw
    ${MM_VPP_COMMAND} ${MM_VPP_LINK_FLAGS}
    MatrixMultiplication_hw.xo -o MatrixMultiplication_hw.xclbin) 
  # Hardware emulation
  add_custom_target(compile_hardware_emulation
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} -c -t hw_emu
    ${MM_VPP_COMMAND} ${MM_VPP_LINK_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMultiplication_hw_emu.xo) 
  add_custom_target(link_hardware_emulation
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} -l -t hw_emu
    ${MM_VPP_COMMAND} ${MM_VPP_LINK_FLAGS}
    MatrixMultiplication_hw_emu.xo -o MatrixMultiplication_hw_emu.xclbin) 
endif()

# Host code to launch kernel
add_executable(RunHardware.exe host/RunHardware.cpp)
target_link_libraries(RunHardware.exe ${Vitis_LIBRARIES} ${BLAS_LIBRARIES} ${Vitis_FLOATING_POINT_LIBRARY} mmkernel)
if(MM_POWER_METER)
  target_link_libraries(RunHardware.exe powermeter)
endif()
