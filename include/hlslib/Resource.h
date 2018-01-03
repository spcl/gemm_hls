/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

// This small header allows injecting HLS resource pragmas from the command
// line. For example, uses the following macro to change the resource of the
// reduction operation:
//
// #ifdef HLSLIB_TREEREDUCE_RESOURCE
// #define HLSLIB_TREEREDUCE_RESOURCE_PRAGMA(var) HLSLIB_RESOURCE_PRAGMA(var,
// HLSLIB_MAP_CORE)
// #else
// #define HLSLIB_TREEREDUCE_RESOURCE_PRAGMA(var)
// #endif
//
// To set the resource, the code should be compiled with e.g.
// -DHLSLIB_TREEREDUCE_RESOURCE=HAddSub_nodsp
//
// This can then be used in the code:
// auto res = a + b;
// HLSLIB_TREEREDUCE_RESOURCE_PRAGMA(res);

#define HLSLIB_STRINGIFY(x) #x
#define HLSLIB_MAKE_PRAGMA(var) _Pragma(HLSLIB_STRINGIFY(var)) 
#define HLSLIB_RESOURCE_PRAGMA(var, _core) HLSLIB_MAKE_PRAGMA(HLS RESOURCE variable=var core=_core)

