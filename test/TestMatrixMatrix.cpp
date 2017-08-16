/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Utility.h"
#include "MatrixMatrix.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

int main() {

  std::vector<Data_t> a(kSize * kSize);
  std::vector<Data_t> b(kSize * kSize);
  std::vector<Data_t> cReference(kSize * kSize, 0);

  std::random_device rd;
  std::default_random_engine rng(rd());
#ifdef MM_INTEGER_TYPE 
  std::uniform_int_distribution<Data_t> dist(1, 10);
#else
  std::uniform_real_distribution<Data_t> dist(0, 1);
#endif

  std::for_each(a.begin(), a.end(),
                [&dist, &rng](Data_t &in) { in = dist(rng); });
  std::for_each(b.begin(), b.end(),
                [&dist, &rng](Data_t &in) { in = dist(rng); });

  const auto bKernel = Pack(b);
  auto cKernel = Pack(cReference);

  Naive<OperatorMap, OperatorReduce>(a.cbegin(), b.cbegin(), cReference.begin(),
                                     kSize, kSize, kSize);
  MatrixMatrix(a.data(), bKernel.data(), cKernel.data());

  const auto cTest = Unpack(cKernel);

  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      const auto testVal = cTest[i * kSize + j];
      const auto refVal = cReference[i * kSize + j];
      const auto diff = std::abs(testVal - refVal);
      if (diff > static_cast<Data_t>(1e-3)) {
        std::cerr << "Mismatch detected at (" << i << ", " << j
                  << "): " << testVal << " vs. " << refVal << "\n";
        return 1;
      }
    }
  }

  std::cout << "Matrix-matrix multiplication successfully verified.\n";

  return 0;
}
