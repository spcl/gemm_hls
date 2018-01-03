/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      April 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#ifndef HLSLIB_SYNTHESIS
#include <thread>
#include <vector>
#include "hlslib/Stream.h"
#endif

// This header provides functionality to simulate dataflow functions that
// include loops in conjunction with Stream.h.
//
// Dataflow functions must be wrapped in the HLSLIB_DATAFLOW_FUNCTION macro,
// which will launch them as a C++ thread when running simulation, but will
// fall back on normal function calls when running synthesis.
//
// The macro HLSLIB_DATAFLOW_FINALIZE must be called before returning from the
// top level function to join the dataflow threads.
// HLSLIB_DATAFLOW_INIT currently has no purpose, but is included for symmetry.
//
// TODO: HLSLIB_DATAFLOW_FUNCTION does not work when calling templated functions
//       with multiple arguments, as it considers the comma a separator between
//       function arguments. Look into alternative implementation, or always use
//       variadic templates if newer Vivado HLS start supporting inlining the
//       called function (it currently fails with "no function body" on the
//       dataflow function in question).

namespace hlslib {

#ifdef HLSLIB_SYNTHESIS
#define HLSLIB_DATAFLOW_INIT()
#define HLSLIB_DATAFLOW_FUNCTION(func, ...) func(__VA_ARGS__)
#define HLSLIB_DATAFLOW_FINALIZE()
#else
namespace {
class _Dataflow {
private:
  inline _Dataflow() {}
  inline ~_Dataflow() { this->Join(); }
public:
  inline static _Dataflow& Get() { // Singleton pattern
    static _Dataflow df;
    return df; 
  }
  template <class Function, typename... Args>
  void AddFunction(Function &&func, Args&&... args) {
    threads_.emplace_back([&](){ func(args...); });
  }
  template <typename T, typename... Args>
  Stream<T>& EmplaceStream(Args&&... args) {
    streams_.emplace_back(std::unique_ptr<Stream<T>>(new Stream<T>(args...)));
    return *static_cast<Stream<T> *>(streams_.back().get());
  }
  inline void Join() {
    for (auto &t : threads_) {
      t.join();
    }
    threads_.clear();
  }
private:
  std::vector<std::thread> threads_{};
  std::vector<std::unique_ptr<_StreamBase>> streams_{};
};
#define HLSLIB_DATAFLOW_INIT() 
#define HLSLIB_DATAFLOW_FUNCTION(func, ...) \
  ::hlslib::_Dataflow::Get().AddFunction(func, __VA_ARGS__)
#define HLSLIB_DATAFLOW_FINALIZE() ::hlslib::_Dataflow::Get().Join();
}
#endif

} // End namespace hlslib
