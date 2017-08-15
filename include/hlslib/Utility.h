/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      December 2016
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

namespace hlslib {

// Constexpr methods for compile-time computations

constexpr unsigned char ConstLog2(unsigned long val) {
  return val == 0 ? 0 : 1 + ConstLog2(val >> 1);
}

// Integer division with ceil instead of floor
template <typename T>
constexpr T CeilDivide(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
constexpr T &min(T &a, T &b) { return (a < b) ? a : b; } 

template <typename T>
constexpr T const &min(T const &a, const T &b) { return (a < b) ? a : b; } 

template <typename T>
constexpr T &max(T &a, T &b) { return (a > b) ? a : b; } 

template <typename T>
constexpr T const &max(T const &a, const T &b) { return (a > b) ? a : b; } 

constexpr signed long abs(const signed long a) {
  return (a < 0) ? -a : a;
}

};
