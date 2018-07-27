# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Created: June 2017

cmake_minimum_required(VERSION 3.0)
project(sdaccel_mm)

# Target options 
set(MM_PART_NAME "xcvu9p-flgb2104-2-i" CACHE STRING "Part name for HLS.")
set(MM_DSA_NAME "xilinx_vcu1525_dynamic_5_0" CACHE STRING
    "DSA string for xocc.")
set(MM_TARGET_CLOCK 200 CACHE STRING
    "Target clock for kernel (<=0 uses DSA default).")
set(MM_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(MM_TMP_DIR /tmp CACHE STRING "Temporary directory for SDx build files")

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_MEMORY_BUS_WIDTH 64 CACHE STRING "Width of memory bus in bytes.")
set(MM_SIZE_N 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_K 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 512 CACHE STRING "Size of matrix dimension.")
set(MM_MEMORY_TILE_SIZE 256 CACHE STRING "Tile size of outer memory tile.")
set(MM_PARALLELISM_N 16 CACHE STRING "Number of parallel compute in N.")
set(MM_PARALLELISM_M 16 CACHE STRING "Number of parallel compute in M.")
set(MM_GRANULARITY_N 1 CACHE STRING "Granularity of processing elements in N.")
set(MM_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULTIPLY_RESOURCE OFF CACHE STRING "")  

# Internal
set(MM_KERNEL_NAME MatrixMultiplicationKernel)
set(MM_ENTRY_FUNCTION MatrixMultiplication)
include(CheckTypeSize)
check_type_size(${MM_DATA_TYPE} MM_KERNEL_WIDTH_N)
math(EXPR MM_KERNEL_WIDTH_M "${MM_KERNEL_WIDTH_N} * ${MM_PARALLELISM_M}")

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
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

# Executables 
add_executable(Stats src/Stats.cpp)

# Software test
add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} mmkernel)
add_test(TestSimulation TestSimulation)
enable_testing()

# Synthesis
set(MM_HLS_PROJECT_NAME "hls")
string(REPLACE ";" " " MM_SRC_SYNTHESIS "${MM_KERNEL_SRC}")
set(MM_ENTRY_FUNCTION "MatrixMultiplicationKernel")
set(MM_SYNTHESIS_FLAGS "-std=c++11 -O3 -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS")
configure_file(scripts/Synthesis.tcl.in Synthesis.tcl)
add_custom_target(synthesis COMMAND ${SDAccel_VIVADO_HLS} -f Synthesis.tcl) 

# Vivado project options
set(DONUT_KERNEL_NAME ${MM_KERNEL_NAME}) 
set(DONUT_KERNEL_DIRECTORY ${CMAKE_BINARY_DIR})
set(DONUT_LIBRARY_NAME "MM")
set(DONUT_VENDOR_NAME SPCL)
set(DONUT_TMP_DIR ${MM_TMP_DIR})
set(DONUT_PART_NAME ${MM_PART_NAME})
set(DONUT_SOURCES_DIRECTORY ${CMAKE_BINARY_DIR}/${MM_HLS_PROJECT_NAME}/${MM_PART_NAME}/syn/verilog/)
if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
  set(DONUT_CALL_PACKAGE_FUNCTION 0)
else()
  set(DONUT_CALL_PACKAGE_FUNCTION 1)
endif() 
configure_file(scripts/config.tcl.in config.tcl)
configure_file(hw/top.v.in top.v)

# SDx kernel wizard custom RTL project
add_custom_target(setup_project COMMAND ${CMAKE_SOURCE_DIR}/scripts/vivado.py setup ${DONUT_KERNEL_NAME} ${SDAccel_VIVADO} ${CMAKE_BINARY_DIR} ${DONUT_TMP_DIR})
add_custom_target(package_kernel COMMAND ${CMAKE_SOURCE_DIR}/scripts/vivado.py package ${DONUT_KERNEL_NAME} ${SDAccel_VIVADO} ${CMAKE_BINARY_DIR} ${DONUT_TMP_DIR})

# Remove Vivado project when running clean
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
             ${DONUT_KERNEL_DIRECTORY}/${DONUT_KERNEL_NAME}_ex)

# Hardware kernel
set(MM_XOCC_COMMAND
  -t hw
  -s
  -O3
  ${MM_XOCC_FLAGS}
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  --kernel ${MM_KERNEL_NAME} 
  --platform ${MM_DSA_NAME}
  --xp prop:kernel.${MM_ENTRY_FUNCTION}.kernel_flags="${MM_SYNTHESIS_FLAGS}"
  --max_memory_ports all)
if(MM_TARGET_CLOCK GREATER 0)
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
      --kernel_frequency ${MM_TARGET_CLOCK})
endif()
if(NOT (MM_DSA_NAME MATCHES "dynamic"))
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
    --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.m00_axi.core.OCL_REGION_0.M00_AXI
    --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.m01_axi.core.OCL_REGION_0.M01_AXI
    --xp misc:map_connect=add.kernel.${MM_KERNEL_NAME}_1.m02_axi.core.OCL_REGION_0.M01_AXI)
else()
  set(MM_XOCC_COMMAND ${MM_XOCC_COMMAND}
    --sp ${MM_KERNEL_NAME}_1.m_axi_gmem0:bank0
    --sp ${MM_KERNEL_NAME}_1.m_axi_gmem1:bank1
    --sp ${MM_KERNEL_NAME}_1.m_axi_gmem1:bank1)
endif()

if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
  add_custom_target(build_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
    ${MM_XOCC_COMMAND} ${MM_XOCC_COMPILE_FLAGS} ${MM_XOCC_LINK_FLAGS} -l
    MatrixMultiplicationKernel.xo -o MatrixMultiplication.xclbin) 
else()
  add_custom_target(compile_from_cpp
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c
    ${MM_XOCC_COMMAND} ${MM_XOCC_LINK_FLAGS}
    ${MM_KERNEL_SRC} -o MatrixMultiplicationKernel.xo) 
  add_custom_target(build_kernel
    COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l
    ${MM_XOCC_COMMAND} ${MM_XOCC_LINK_FLAGS}
    MatrixMultiplicationKernel.xo -o MatrixMultiplication.xclbin) 
endif()

# Host code to launch kernel
add_executable(RunHardware host/RunHardware.cpp)
target_link_libraries(RunHardware ${SDAccel_LIBRARIES} ${BLAS_LIBRARIES} mmkernel)
