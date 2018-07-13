/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
#include "MatrixMatrix.h"
#ifdef MM_HAS_BLAS
#include "cblas.h"
#endif

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

inline std::vector<MemoryPack_t> Pack(std::vector<Data_t> const &in) {
  std::vector<MemoryPack_t> result(in.size() / kMemoryWidth);
  for (int i = 0, iMax = in.size() / kMemoryWidth; i < iMax; ++i) {
    result[i].Pack(&in[i * kMemoryWidth]);;
  }
  return result;
}

inline std::vector<Data_t> Unpack(std::vector<MemoryPack_t> const &in) {
  std::vector<Data_t> result(in.size() * kMemoryWidth);
  for (int i = 0, iMax = in.size(); i < iMax; ++i) {
    in[i].Unpack(&result[i * kMemoryWidth]);
  }
  return result;
}

// Fallback
template <typename T, class OperatorMap, class OperatorReduce>
void CallBLAS(T const *a, T const *b, T *c) {
  Naive<OperatorMap, OperatorReduce>(a, b, c, kSizeN, kSizeK, kSizeM);
}

#ifdef MM_HAS_BLAS
template <>
void CallBLAS<float, hlslib::op::Multiply<float>, hlslib::op::Add<float>>(
    float const *a, float const *b, float *c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSizeN, kSizeM, kSizeK,
              1.0, a, kSizeK, b, kSizeM, 0.0, c, kSizeM);
}
template <>
void CallBLAS<double, hlslib::op::Multiply<double>, hlslib::op::Add<double>>(
    double const *a, double const *b, double *c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSizeN, kSizeM, kSizeK,
              1.0, a, kSizeK, b, kSizeM, 0.0, c, kSizeM);
}
#endif

inline void ReferenceImplementation(Data_t const *a, Data_t const *b,
                                    Data_t *c, bool useCache) {
  std::string goldenFileName;
  if (useCache) {
    std::stringstream ss;
    ss << kGoldenDir << "gemm_n" << kSizeN << "_m" << kSizeK << "_p" << kSizeM
       << "_s" << kSeed << ".dat";
    goldenFileName = ss.str();
    std::ifstream goldenFile(goldenFileName,
                             std::ios_base::in | std::ios_base::binary);
    if (goldenFile.good()) {
      std::cout << "Using cached golden result." << std::endl;
      goldenFile.read(reinterpret_cast<char *>(&c[0]),
                      kSizeN * kSizeM * sizeof(Data_t));
      return;
    } else {
      std::cout << "No cached result found.\n" << std::flush;
    }
  }

  std::cout << "Running host implementation..." << std::flush;
  CallBLAS<Data_t, OperatorMap, OperatorReduce>(a, b, c);
  std::cout << " Done.\n";

  if (useCache) {
    std::ofstream goldenFileOut(goldenFileName,
                                std::ios_base::out | std::ios_base::binary);
  }
}
