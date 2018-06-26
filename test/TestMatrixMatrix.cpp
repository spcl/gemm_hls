/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
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

  std::vector<Data_t> a(kSizeN * kSizeK);
  std::vector<Data_t> b(kSizeK * kSizeM);
  std::vector<Data_t> cReference(kSizeN * kSizeM, 0);

  std::default_random_engine rng(kSeed);
  typename std::conditional<
      std::is_integral<Data_t>::value, std::uniform_int_distribution<unsigned long>,
      std::uniform_real_distribution<double>>::type dist(1, 10);

  std::for_each(a.begin(), a.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
  std::for_each(b.begin(), b.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });

  // const auto aKernel = Pack(a);
  const auto bKernel = Pack(b);
  auto cKernel = Pack(cReference);
  const std::vector<Data_t> aKernel(a);

  ReferenceImplementation(a.data(), b.data(), cReference.data());

  std::cout << "Running hardware emulation..." << std::flush;
  MatrixMatrix(aKernel.data(), bKernel.data(), cKernel.data());
  std::cout << " Done.\n";

  const auto cTest = Unpack(cKernel);

  std::cout << "Verifying results..." << std::endl;
  for (int i = 0; i < kSizeN; ++i) {
    for (int j = 0; j < kSizeM; ++j) {
      const auto testVal = cTest[i * kSizeM + j];
      const auto refVal = cReference[i * kSizeM + j];
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
