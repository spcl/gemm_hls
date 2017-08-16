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

std::vector<MemoryPack_t> Pack(std::vector<Data_t> const &in) {
  std::vector<MemoryPack_t> result(in.size() / kMemoryWidth);
  for (int i = 0, iMax = in.size() / kMemoryWidth; i < iMax; ++i) {
    KernelPack_t pack[kKernelPerMemory];
    for (int j = 0; j < kKernelPerMemory; ++j) {
      pack[j] = KernelPack_t(&in[i * kMemoryWidth + j * kKernelWidth]);
    }
    result[i] = MemoryPack_t(pack);
  }
  return result;
}

std::vector<Data_t> Unpack(std::vector<MemoryPack_t> const &in) {
  std::vector<Data_t> result(in.size() * kMemoryWidth);
  for (int i = 0, iMax = in.size(); i < iMax; ++i) {
    const MemoryPack_t mem = in[i];
    for (int j = 0; j < kKernelPerMemory; ++j) {
      const KernelPack_t pack = mem[j];
      for (int k = 0; k < kKernelWidth; ++k) {
        result[i * kMemoryWidth + j * kKernelWidth + k] = pack[k]; 
      }
    }
  }
  return result;
}
