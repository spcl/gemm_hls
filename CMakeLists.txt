# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Created: June 2017

cmake_minimum_required(VERSION 2.8)
project(sdaccel_mm C CXX)

# Target options 
set(MM_PART_NAME "xcku115-flvb2104-2-e" CACHE STRING "Part name for HLS.")
set(MM_DSA_NAME "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0" CACHE STRING
    "DSA string for xocc.")
set(MM_TARGET_CLOCK 200 CACHE STRING "Target clock for kernel (<=0 uses DSA default).")
set(MM_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(MM_SYNTHESIS_FLAGS "" CACHE STRING "Extra flags for synthesis")
set(MM_XOCC_FLAGS "" CACHE STRING "Extra flags for both compile and link steps of the xocc flow")
set(MM_XOCC_COMPILE_FLAGS "" CACHE STRING "Extra flags for xocc compile step")
set(MM_XOCC_LINK_FLAGS "" CACHE STRING "Extra flags for xocc link step")

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

# Check if BLAS is available
find_package(BLAS)
if(BLAS_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMM_HAS_BLAS")
else()
  set(BLAS_LIBRARIES)
endif()

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -std=c++11 -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS")

# Allow specifying the floating point unit for adders and multipliers
if(MM_ADD_RESOURCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
endif() 
if(MM_MULTIPLY_RESOURCE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_OPERATOR_MULTIPLY_RESOURCE=${MM_MULTIPLY_RESOURCE}") 
endif() 

# Hardware configuration header for HLS
set(MM_GOLDEN_DIR ${CMAKE_SOURCE_DIR}/golden/)
configure_file(include/Config.h.in Config.h)

# C++ source code 
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/MatrixMatrix.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

# Executables 
add_executable(Stats src/Stats.cpp)

# Software test
add_executable(TestMatrixMatrix test/TestMatrixMatrix.cpp)
target_link_libraries(TestMatrixMatrix ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} mmkernel)
add_test(TestMatrixMatrix TestMatrixMatrix)
enable_testing()

# Synthesis
set(MM_PROJECT_NAME "MatrixMatrix")
string(REPLACE ";" " " MM_SRC_SYNTHESIS "${MM_KERNEL_SRC}")
set(MM_ENTRY_FUNCTION "MatrixMatrix")
configure_file(scripts/Synthesis.tcl.in SynthesizeMatrixMatrix.tcl)
add_custom_target(synthesis COMMAND ${SDAccel_VIVADO_HLS} -f SynthesizeMatrixMatrix.tcl) 

# Hardware kernel
set(MM_XOCC_COMMAND
  -t hw
  -s
  -O3
  ${MM_XOCC_FLAGS}
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  --kernel MatrixMatrix
  --platform ${MM_DSA_NAME}
  --xp prop:kernel.${MM_ENTRY_FUNCTION}.kernel_flags="${MM_SYNTHESIS_FLAGS}"
  # Matrices A and C use the same interface
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI
  --xp misc:map_connect=add.kernel.${MM_ENTRY_FUNCTION}_1.M_AXI_GMEM2.core.OCL_REGION_0.M00_AXI
  --max_memory_ports all)
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
      --kernel_frequency ${MM_TARGET_CLOCK})
endif()

if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
  add_custom_target(build_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
    ${MM_XOCC_COMMAND} ${MM_XOCC_COMPILE_FLAGS} ${MM_XOCC_LINK_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMatrix.xclbin) 
else()
  add_custom_target(compile_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c
    ${MM_XOCC_COMMAND} ${MM_XOCC_COMPILE_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMatrix.xo) 
  add_custom_target(link_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l
    ${MM_XOCC_COMMAND} ${MM_XOCC_LINK_FLAGS}
    MatrixMatrix.xo -o MatrixMatrix.xclbin) 
endif()

add_executable(RunMatrixMatrix host/RunMatrixMatrix.cpp)
target_link_libraries(RunMatrixMatrix ${SDAccel_LIBRARIES} mmkernel ${BLAS_LIBRARIES}) 
