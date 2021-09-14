/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
#include "MatrixMultiplication.h"
#include "hlslib/xilinx/OpenCL.h"
#ifdef MM_HAS_BLAS
#include "cblas.h"
#endif

template <class OperatorMap, class OperatorReduce, class IteratorRead,
          class IteratorWrite>
void Naive(IteratorRead aBegin, IteratorRead bBegin, IteratorWrite cBegin,
           int sizeN, int sizeK, int sizeM) {
  using TIn = typename std::iterator_traits<IteratorRead>::value_type;
  using TOut = typename std::iterator_traits<IteratorWrite>::value_type;
  static_assert(std::is_same<TIn, TOut>::value,
                "Input and output types must be identical.");
  // NxM * MxP = NxP
  for (int n = 0; n < sizeN; ++n) {
    for (int m = 0; m < sizeM; ++m) {
      TOut acc = OperatorReduce::identity();
      for (int k = 0; k < sizeK; ++k) {
#ifndef MM_TRANSPOSED_A
        const auto elemA = aBegin[n * sizeK + k];
#else
        const auto elemA = aBegin[k * sizeN + n];
#endif
        const auto elemB = bBegin[k * sizeM + m];
        acc = OperatorReduce::Apply(acc, OperatorMap::Apply(elemA, elemB));
      }
      cBegin[n * sizeM + m] = acc;
    }
  }
}

template <int width, class Container>
inline auto Pack(Container const &in) {
  std::vector<
      hlslib::DataPack<Data_t, width>,
      hlslib::ocl::AlignedAllocator<hlslib::DataPack<Data_t, width>, 4096>>
      result(in.size() / width);
  for (int i = 0, iMax = in.size() / width; i < iMax; ++i) {
    result[i].Pack(&in[i * width]);
  }
  return result;
}

template <int width, class Container>
inline auto Unpack(Container const &in) {
  std::vector<Data_t> result(in.size() * width);
  for (int i = 0, iMax = in.size(); i < iMax; ++i) {
    in[i].Unpack(&result[i * width]);
  }
  return result;
}

// Fallback
template <typename T, class OperatorMap, class OperatorReduce>
void CallBLAS(T const *a, T const *b, T *c, const unsigned size_n,
              const unsigned size_k, const unsigned size_m) {
  std::cout
      << "WARNING: BLAS not available, so I'm falling back on a naive "
         "implementation. This will take a long time for large matrix sizes.\n"
      << std::flush;
  Naive<OperatorMap, OperatorReduce>(a, b, c, size_n, size_k, size_m);
}

#ifdef MM_HAS_BLAS
template <>
void CallBLAS<float, hlslib::op::Multiply<float>, hlslib::op::Add<float>>(
    float const *a, float const *b, float *c, const unsigned size_n,
    const unsigned size_k, const unsigned size_m) {
  std::cout << "Running BLAS...\n" << std::flush;
#ifndef MM_TRANSPOSED_A
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size_n, size_m, size_k,
              1.0, a, size_k, b, size_m, 0.0, c, size_m);
#else
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, size_n, size_m, size_k,
              1.0, a, size_k, b, size_m, 0.0, c, size_m);
#endif
}
template <>
void CallBLAS<double, hlslib::op::Multiply<double>, hlslib::op::Add<double>>(
    double const *a, double const *b, double *c, const unsigned size_n,
    const unsigned size_k, const unsigned size_m) {
  std::cout << "Running BLAS...\n" << std::flush;
#ifndef MM_TRANSPOSED_A
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size_n, size_m, size_k,
              1.0, a, size_k, b, size_m, 0.0, c, size_m);
#else
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, size_n, size_m, size_k,
              1.0, a, size_k, b, size_m, 0.0, c, size_m);
#endif
}
#endif

inline void ReferenceImplementation(Data_t const *a, Data_t const *b, Data_t *c,
                                    const unsigned size_n,
                                    const unsigned size_k,
                                    const unsigned size_m) {
  CallBLAS<Data_t, OperatorMap, OperatorReduce>(a, b, c, size_n, size_k,
                                                size_m);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, signed long>::type
make_signed(T val) {
  return static_cast<signed long>(val);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type make_signed(
    T val) {
  return val;
}

template <typename T>
typename std::enable_if<std::is_same<T, half>::value, half>::type make_signed(
    half val) {
  return val;
}
