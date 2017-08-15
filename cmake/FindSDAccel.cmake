# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: March 2017 
# This software is copyrighted under the BSD 3-Clause License. 
#
# Once done this will define:
#   SDAccel_FOUND - Indicates whether SDAccel was found.
#   SDAccel_INCLUDE_DIRS - Include directories for HLS. 
#   SDAccel_LIBRARIES - Libraries required for host side code. 
#   SDAccel_XOCC - Path to the xocc executable.
#   SDAccel_VIVADO_HLS - Path to the Vivado HLS executable shipped with SDAccel. 
#
# To specify the location of SDAccel or to force this script to use a specific
# version, set the variable SDACCEL_ROOT_DIR to the root directory of the
# desired SDAccel installation.

if(NOT DEFINED SDACCEL_ROOT_DIR)

  find_path(SDACCEL_SEARCH_PATH xocc 
            PATHS ENV XILINX_OPENCL ENV XILINX_SDACCEL
						PATH_SUFFIXES bin)
  get_filename_component(SDACCEL_ROOT_DIR ${SDACCEL_SEARCH_PATH} DIRECTORY) 
  mark_as_advanced(SDACCEL_SEARCH_PATH)

else()

  message(STATUS "Using user defined SDAccel directory \"${SDACCEL_ROOT_DIR}\".")

endif()

# Check if all the necessary components are present. We want to ensure that we
# use the tools bundled together, so we don't use find_path again. 

find_program(SDAccel_XOCC xocc PATHS ${SDACCEL_ROOT_DIR}/bin NO_DEFAULT_PATH)

find_program(SDAccel_VIVADO_HLS vivado_hls
             PATHS ${SDACCEL_ROOT_DIR}/Vivado_HLS/bin NO_DEFAULT_PATH)

find_path(SDAccel_HLS_INCLUDE_DIR hls_stream.h
          PATHS ${SDACCEL_ROOT_DIR}/Vivado_HLS/include NO_DEFAULT_PATH)
mark_as_advanced(SDAccel_HLS_INCLUDE_DIR)

# Currently only x86 support

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")

  set(SDACCEL_RUNTIME_LIBS ${SDACCEL_ROOT_DIR}/runtime/lib/x86_64)
  mark_as_advanced(SDACCEL_RUNTIME_LIBS)

  find_library(SDAccel_LIBXILINXOPENCL xilinxopencl PATHS ${SDACCEL_RUNTIME_LIBS}
               NO_DEFAULT_PATH)
  mark_as_advanced(SDAccel_LIBXILINXOPENCL)

  find_library(SDAccel_LIBFLOATINGPOINT Ip_floating_point_v7_0_bitacc_cmodel
               PATHS ${SDACCEL_ROOT_DIR}/Vivado_HLS/lnx64/tools/fpo_v7_0)
  mark_as_advanced(SDAccel_LIBFLOATINGPOINT)

  # Only succeed if all libraries were found
  if(SDAccel_LIBXILINXOPENCL AND SDAccel_LIBFLOATINGPOINT)
    set(SDAccel_LIBRARIES ${SDAccel_LIBXILINXOPENCL} ${SDAccel_LIBFLOATINGPOINT}
        CACHE STRING "SDAccel runtime libraries.")
  endif()

  if(EXISTS ${SDACCEL_RUNTIME_LIBS}/libstdc++.so)
    message(WARNING "Found libstdc++.so in ${SDACCEL_RUNTIME_LIBS}. This may break compilation for newer compilers. Remove from path to ensure that your native libstdc++.so is used.") 
  endif()

  find_path(SDAccel_OPENCL_INCLUDE_DIR opencl.h
            PATHS ${SDACCEL_ROOT_DIR}/runtime/include
            PATH_SUFFIXES 1_1/CL 1_2/CL 2_0/CL
            NO_DEFAULT_PATH)
  get_filename_component(SDAccel_OPENCL_INCLUDE_DIR ${SDAccel_OPENCL_INCLUDE_DIR} DIRECTORY) 
  mark_as_advanced(SDAccel_OPENCL_INCLUDE_DIR)

  # Only succeed if both include paths were found
  if(SDAccel_HLS_INCLUDE_DIR AND SDAccel_OPENCL_INCLUDE_DIR)
    set(SDAccel_INCLUDE_DIRS ${SDAccel_HLS_INCLUDE_DIR}
        ${SDAccel_OPENCL_INCLUDE_DIR} 
        CACHE STRING "SDAccel include directories.")
  endif()

else()

  message(WARNING "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")

endif()

include(FindPackageHandleStandardArgs)
# Handle the QUIETLY and REQUIRED arguments and set SDAccelHLS_FOUND to TRUE
# if all listed variables were found.
find_package_handle_standard_args(SDAccel DEFAULT_MSG
  SDAccel_XOCC SDAccel_VIVADO_HLS SDAccel_INCLUDE_DIRS
  SDAccel_LIBRARIES)
