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

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_KERNEL_WIDTH 4 CACHE STRING "Kernel data path width in number of operands.")
set(MM_SIZE_N 256 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 256 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_P 256 CACHE STRING "Size of matrix dimension.")
set(MM_TILE_SIZE_N 16 CACHE STRING "Tile size in N.")
set(MM_TILE_SIZE_P 64 CACHE STRING "Tile size in P.")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)

include_directories(include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

configure_file(include/Config.h.in Config.h)

enable_testing()

set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/MatrixMatrix.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

add_executable(Stats src/Stats.cpp)

add_executable(TestMatrixMatrix test/TestMatrixMatrix.cpp)
target_link_libraries(TestMatrixMatrix ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} mmkernel)
add_test(TestMatrixMatrix TestMatrixMatrix)

set(MM_PROJECT_NAME "MatrixMatrix")
string(REPLACE ";" " " MM_SRC_SYNTHESIS "${MM_KERNEL_SRC}")
set(MM_ENTRY_FUNCTION "MatrixMatrix")
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_TARGET_TIMING "${MM_TARGET_CLOCK}")
else()
  set(MM_TARGET_TIMING "5")
endif()
configure_file(scripts/Synthesis.tcl.in SynthesizeMatrixMatrix.tcl)
add_custom_target(synthesize_matrix_matrix
  COMMAND ${SDAccel_VIVADO_HLS} -f SynthesizeMatrixMatrix.tcl) 
if(SDAccel_FOUND)
  set(MM_XOCC_COMMAND
    -t hw
    -O3
    -s
    -o MatrixMatrix.xclbin
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_BINARY_DIR}
    ${MM_KERNEL_SRC}
    --kernel MatrixMatrix
    --xdevice ${MM_DSA_NAME}
    --xp prop:kernel.${MM_ENTRY_FUNCTION}.kernel_flags="${CMAKE_CXX_FLAGS} -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS"
    # Matrices A and C use the same interface
    --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM0.core.OCL_REGION_0.M01_AXI
    --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM1.core.OCL_REGION_0.M00_AXI
    --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM2.core.OCL_REGION_0.M01_AXI
    --max_memory_ports all)
  if(MM_TARGET_CLOCK GREATER 0)
    set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
        --kernel_frequency ${MM_TARGET_CLOCK})
  endif()
  add_custom_target(kernel_matrix_matrix
    COMMAND ${SDAccel_XOCC} ${MM_XOCC_COMMAND}) 
  add_executable(RunMatrixMatrix.exe host/RunMatrixMatrix.cpp)
  target_link_libraries(RunMatrixMatrix.exe ${SDAccel_LIBRARIES} mmkernel) 
endif()
