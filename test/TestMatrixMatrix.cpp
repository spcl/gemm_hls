/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Utility.h"
#include "MatrixMatrix.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

int main() {

  std::vector<Data_t> a(kSizeN * kSizeM);
  std::vector<Data_t> b(kSizeM * kSizeP);
  std::vector<Data_t> cReference(kSizeN * kSizeP, 0);

  std::random_device rd;
  std::default_random_engine rng(rd());
  typename std::conditional<
      std::is_integral<Data_t>::value, std::uniform_int_distribution<Data_t>,
      std::uniform_real_distribution<Data_t>>::type dist(1, 10);

  std::for_each(a.begin(), a.end(),
                [&dist, &rng](Data_t &in) { in = dist(rng); });
  std::for_each(b.begin(), b.end(),
                [&dist, &rng](Data_t &in) { in = dist(rng); });

  const auto aKernel = Pack(a);
  const auto bKernel = Pack(b);
  auto cKernel = Pack(cReference);

  Naive<OperatorMap, OperatorReduce>(a.cbegin(), b.cbegin(), cReference.begin(),
                                     kSizeN, kSizeM, kSizeP);
  MatrixMatrix(aKernel.data(), bKernel.data(), cKernel.data());

  const auto cTest = Unpack(cKernel);

  for (int i = 0; i < kSizeN; ++i) {
    for (int j = 0; j < kSizeP; ++j) {
      const auto testVal = cTest[i * kSizeP + j];
      const auto refVal = cReference[i * kSizeP + j];
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
