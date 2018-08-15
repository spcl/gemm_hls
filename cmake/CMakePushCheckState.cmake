#.rst:
# CMakePushCheckState
# -------------------
#
#
#
# This module defines three macros: CMAKE_PUSH_CHECK_STATE()
# CMAKE_POP_CHECK_STATE() and CMAKE_RESET_CHECK_STATE() These macros can
# be used to save, restore and reset (i.e., clear contents) the state of
# the variables CMAKE_REQUIRED_FLAGS, CMAKE_REQUIRED_DEFINITIONS,
# CMAKE_REQUIRED_LIBRARIES and CMAKE_REQUIRED_INCLUDES used by the
# various Check-files coming with CMake, like e.g.
# check_function_exists() etc.  The variable contents are pushed on a
# stack, pushing multiple times is supported.  This is useful e.g.  when
# executing such tests in a Find-module, where they have to be set, but
# after the Find-module has been executed they should have the same
# value as they had before.
#
# CMAKE_PUSH_CHECK_STATE() macro receives optional argument RESET.
# Whether it's specified, CMAKE_PUSH_CHECK_STATE() will set all
# CMAKE_REQUIRED_* variables to empty values, same as
# CMAKE_RESET_CHECK_STATE() call will do.
#
# Usage:
#
# ::
#
#    cmake_push_check_state(RESET)
#    set(CMAKE_REQUIRED_DEFINITIONS -DSOME_MORE_DEF)
#    check_function_exists(...)
#    cmake_reset_check_state()
#    set(CMAKE_REQUIRED_DEFINITIONS -DANOTHER_DEF)
#    check_function_exists(...)
#    cmake_pop_check_state()

#=============================================================================
# Copyright 2006-2011 Alexander Neundorf, <neundorf@kde.org>
#
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2016 Kitware, Inc.
# Copyright 2000-2011 Insight Software Consortium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ------------------------------------------------------------------------------
#
# The above copyright and license notice applies to distributions of
# CMake in source and binary form.  Some source files contain additional
# notices of original copyright by their contributors; see each source
# for details.  Third-party software packages supplied with CMake under
# compatible licenses provide their own copyright notices documented in
# corresponding subdirectories.
#
# ------------------------------------------------------------------------------
#
# CMake was initially developed by Kitware with the following sponsorship:
#
#  * National Library of Medicine at the National Institutes of Health
#    as part of the Insight Segmentation and Registration Toolkit (ITK).
#
#  * US National Labs (Los Alamos, Livermore, Sandia) ASC Parallel
#    Visualization Initiative.
#
#  * National Alliance for Medical Image Computing (NAMIC) is funded by the
#    National Institutes of Health through the NIH Roadmap for Medical Research,
#    Grant U54 EB005149.
#
#  * Kitware, Inc.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================


macro(CMAKE_RESET_CHECK_STATE)

   set(CMAKE_REQUIRED_INCLUDES)
   set(CMAKE_REQUIRED_DEFINITIONS)
   set(CMAKE_REQUIRED_LIBRARIES)
   set(CMAKE_REQUIRED_FLAGS)
   set(CMAKE_REQUIRED_QUIET)

endmacro()

macro(CMAKE_PUSH_CHECK_STATE)

   if(NOT DEFINED _CMAKE_PUSH_CHECK_STATE_COUNTER)
      set(_CMAKE_PUSH_CHECK_STATE_COUNTER 0)
   endif()

   math(EXPR _CMAKE_PUSH_CHECK_STATE_COUNTER "${_CMAKE_PUSH_CHECK_STATE_COUNTER}+1")

   set(_CMAKE_REQUIRED_INCLUDES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}    ${CMAKE_REQUIRED_INCLUDES})
   set(_CMAKE_REQUIRED_DEFINITIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER} ${CMAKE_REQUIRED_DEFINITIONS})
   set(_CMAKE_REQUIRED_LIBRARIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}   ${CMAKE_REQUIRED_LIBRARIES})
   set(_CMAKE_REQUIRED_FLAGS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}       ${CMAKE_REQUIRED_FLAGS})
   set(_CMAKE_REQUIRED_QUIET_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}       ${CMAKE_REQUIRED_QUIET})

   if (ARGC GREATER 0 AND ARGV0 STREQUAL "RESET")
      cmake_reset_check_state()
   endif()

endmacro()

macro(CMAKE_POP_CHECK_STATE)

# don't pop more than we pushed
   if("${_CMAKE_PUSH_CHECK_STATE_COUNTER}" GREATER "0")

      set(CMAKE_REQUIRED_INCLUDES    ${_CMAKE_REQUIRED_INCLUDES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
      set(CMAKE_REQUIRED_DEFINITIONS ${_CMAKE_REQUIRED_DEFINITIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
      set(CMAKE_REQUIRED_LIBRARIES   ${_CMAKE_REQUIRED_LIBRARIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
      set(CMAKE_REQUIRED_FLAGS       ${_CMAKE_REQUIRED_FLAGS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
      set(CMAKE_REQUIRED_QUIET       ${_CMAKE_REQUIRED_QUIET_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})

      math(EXPR _CMAKE_PUSH_CHECK_STATE_COUNTER "${_CMAKE_PUSH_CHECK_STATE_COUNTER}-1")
   endif()

endmacro()
