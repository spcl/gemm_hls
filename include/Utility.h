/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <cstddef>
#include <iterator>
#include <vector>
#include "MatrixMatrix.h"

template <class OperatorMap, class OperatorReduce, class IteratorRead,
          class IteratorWrite>
void Naive(IteratorRead aBegin, IteratorRead bBegin, IteratorWrite cBegin,
           int sizeN, int sizeM, int sizeP) {
  using TIn = typename std::iterator_traits<IteratorRead>::value_type;
  using TOut = typename std::iterator_traits<IteratorWrite>::value_type;
  static_assert(std::is_same<TIn, TOut>::value,
                "Input and output types must be identical.");
  // NxM * MxP = NxP
  for (int n = 0; n < sizeN; ++n) {
    for (int p = 0; p < sizeP; ++p) {
      TOut acc = OperatorReduce::identity();
      for (int m = 0; m < sizeM; ++m) {
        const auto elemA = aBegin[n * sizeM + m];
        const auto elemB = bBegin[m * sizeP + p];
        acc = OperatorReduce::Apply(acc, OperatorMap::Apply(elemA, elemB));
      }
      cBegin[n * sizeP + p] = acc; 
    }
  }
}

template <typename T, int width>
std::vector<hlslib::DataPack<T, width>> Pack(std::vector<T> const &in) {
  std::vector<hlslib::DataPack<T, width>> result(in.size() / width);
  for (int i = 0, iMax = in.size() / width; i < iMax; ++i) {
    result[i] = hlslib::DataPack<T, width>(&in[i * width]);
  }
  return result;
}

template <typename T, int width>
std::vector<T> Unpack(std::vector<hlslib::DataPack<T, width>> const &in) {
  std::vector<T> result(in.size() * width);
  for (int i = 0, iMax = in.size(); i < iMax; ++i) {
    for (int j = 0; j < width; ++j) {
      result[i * width + j] = in[i][j];
    }
  }
  return result;
}
