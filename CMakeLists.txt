# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: June 2017

cmake_minimum_required(VERSION 2.8)
project(sdaccel_mm C CXX)

# Target options 
set(MM_PART_NAME "xcku115-flvb2104-2-e" CACHE STRING "Part name for HLS.")
set(MM_DSA_NAME "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0" CACHE STRING
    "DSA string for xocc.")
set(MM_TARGET_CLOCK 250 CACHE STRING "Target clock for kernel (<=0 uses DSA default).")
set(MM_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(MM_USE_SIMPLE OFF CACHE STRING "Use simple version of the GEMM kernel.")
set(MM_SINGLE_SLR ON CACHE STRING "Only place and route on a single SLR of the KU115")

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_KERNEL_WIDTH 4 CACHE STRING "Kernel data path width in number of operands.")
set(MM_SIZE_N 256 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 256 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_P 256 CACHE STRING "Size of matrix dimension.")
set(MM_TILE_SIZE_N 16 CACHE STRING "Tile size in N.")
set(MM_TILE_SIZE_P 128 CACHE STRING "Tile size in P.")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULTIPLY_RESOURCE OFF CACHE STRING "")  

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Allow specifying the floating point unit for adders and multipliers
if(MM_ADD_RESOURCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
endif() 
if(MM_MULTIPLY_RESOURCE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_MULTIPLY_RESOURCE=${MM_MULTIPLY_RESOURCE}") 
endif() 

# Hardware configuration header for HLS
configure_file(include/Config.h.in Config.h)

# Restrict kernel to single SLR
configure_file(scripts/O_opt_design_pre.tcl data/sdx/xocc/O_opt_design_pre.tcl) 
configure_file(scripts/optMap.xml data/sdx/xocc/optMap.xml) 

# Allow using simple kernel
if(NOT MM_USE_SIMPLE)
  set(MM_KERNEL_SRC
      ${CMAKE_SOURCE_DIR}/kernel/MatrixMatrix.cpp
      ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
else()
  set(MM_KERNEL_SRC
      ${CMAKE_SOURCE_DIR}/kernel/MatrixMatrixSimple.cpp
      ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
endif()
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

# Other exeutables
add_executable(Stats src/Stats.cpp)

# Software test
add_executable(TestMatrixMatrix test/TestMatrixMatrix.cpp)
target_link_libraries(TestMatrixMatrix ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} mmkernel)
add_test(TestMatrixMatrix TestMatrixMatrix)
enable_testing()

# Synthesis
set(MM_PROJECT_NAME "MatrixMatrix")
string(REPLACE ";" " " MM_SRC_SYNTHESIS "${MM_KERNEL_SRC}")
set(MM_ENTRY_FUNCTION "MatrixMatrix")
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_TARGET_TIMING "${MM_TARGET_CLOCK}")
else()
  set(MM_TARGET_TIMING "5")
endif()
configure_file(scripts/Synthesis.tcl.in SynthesizeMatrixMatrix.tcl)
add_custom_target(synthesis COMMAND ${SDAccel_VIVADO_HLS} -f SynthesizeMatrixMatrix.tcl) 

# Hardware kernel
set(MM_XOCC_COMMAND
  -t hw
  -s
  -o MatrixMatrix.xclbin
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  ${MM_KERNEL_SRC}
  --kernel MatrixMatrix
  --platform ${MM_DSA_NAME}
  --xp prop:kernel.${MM_ENTRY_FUNCTION}.kernel_flags="${CMAKE_CXX_FLAGS} -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS"
  # Matrices A and C use the same interface
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM2.core.OCL_REGION_0.M00_AXI
  --max_memory_ports all)
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
      --kernel_frequency ${MM_TARGET_CLOCK})
endif()
if(MM_SINGLE_SLR)
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND} -O4)
else()
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND} -O3)
endif()
add_custom_target(build_kernel
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} ${MM_XOCC_COMMAND}) 
add_executable(RunMatrixMatrix.exe host/RunMatrixMatrix.cpp)
target_link_libraries(RunMatrixMatrix.exe ${SDAccel_LIBRARIES} mmkernel) 
