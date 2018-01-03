/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      May 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

// This header implementations reduction using a binary tree, allowing a fully
// pipelined reduction of a static sized array.
// Using the reduction requires passing an operator that implements the
// functions "Reduce" and "identity", as in the examples given below.

namespace hlslib {

namespace { // Internals

template <typename T, class Operator, int width>
struct TreeReduceImplementation {
  static T f(T arr[width]) {
    #pragma HLS INLINE
    static constexpr int halfWidth = width / 2;
    static constexpr int reducedSize = halfWidth + width % 2;
    T reduced[reducedSize];
    #pragma HLS ARRAY_PARTITION variable=reduced complete
  Reduction:
    for (int i = 0; i < halfWidth; ++i) {
      #pragma HLS UNROLL
      reduced[i] = Operator::Apply(arr[i * 2], arr[i * 2 + 1]);
    }
    if (halfWidth != reducedSize) {
      reduced[reducedSize - 1] = arr[width - 1];
    }
    return TreeReduceImplementation<T, Operator, reducedSize>::f(reduced);
  }
private:
  TreeReduceImplementation() = delete;
  ~TreeReduceImplementation() = delete;
};

template <typename T, class Operator>
struct TreeReduceImplementation<T, Operator, 2> {
  static T f(T arr[2]) {
    #pragma HLS INLINE
    return Operator::Apply(arr[0], arr[1]);
  }
private:
  TreeReduceImplementation() = delete;
  ~TreeReduceImplementation() = delete;
};

template <typename T, class Operator>
struct TreeReduceImplementation<T, Operator, 1> {
  static T f(T arr[1]) {
    #pragma HLS INLINE
    return arr[0];
  }
private:
  TreeReduceImplementation() = delete;
  ~TreeReduceImplementation() = delete;
};

template <typename T, class Operator>
struct TreeReduceImplementation<T, Operator, 0> {
  static constexpr T f(T *) { 
    return Operator::identity();
  }
private:
  TreeReduceImplementation() = delete;
  ~TreeReduceImplementation() = delete;
};

} // End anonymous namespace

/// Reduction entry function.
template <typename T, class Operator, int width>
T TreeReduce(T arr[width]) {
  #pragma HLS PIPELINE
  return TreeReduceImplementation<T, Operator, width>::f(arr);
}

} // End namespace hlslib
