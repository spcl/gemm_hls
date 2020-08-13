/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include "MatrixMultiplication.h"
#include "Utility.h"

int main(int argc, char **argv) {
#ifdef MM_DYNAMIC_SIZES
  if (argc < 4 || argc > 4) {
    std::cerr << "Usage: ./TestSimulation N K M" << std::endl;
    return 1;
  }
  const unsigned size_n = std::stoul(argv[1]);
  const unsigned size_k = std::stoul(argv[2]);
  const unsigned size_m = std::stoul(argv[3]);
  if (size_k % kMemoryWidthK != 0) {
    std::cerr << "K must be divisable by memory width." << std::endl;
    return 1;
  }
#ifndef MM_TRANSPOSED_A
  if (size_k % kTransposeWidth != 0) {
    std::cerr << "K must be divisable by the transpose width." << std::endl;
    return 1;
  }
#endif
  if (size_m % kMemoryWidthM != 0) {
    std::cerr << "M must be divisable by memory width." << std::endl;
    return 1;
  }
#else
  constexpr auto size_n = kSizeN;
  constexpr auto size_k = kSizeK;
  constexpr auto size_m = kSizeM;
#endif

  std::vector<Data_t> a(size_n * size_k);
  std::vector<Data_t> b(size_k * size_m);
  std::vector<Data_t> cReference(size_n * size_m, 0);

  std::default_random_engine rng(kSeed);
  typename std::conditional<std::is_integral<Data_t>::value,
                            std::uniform_int_distribution<unsigned long>,
                            std::uniform_real_distribution<double>>::type
      dist(1, 10);

  std::for_each(a.begin(), a.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
  std::for_each(b.begin(), b.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });

  const auto aKernel = Pack<kMemoryWidthA>(a);
  const auto bKernel = Pack<kMemoryWidthM>(b);
  auto cKernel = Pack<kMemoryWidthM>(cReference);

  ReferenceImplementation(a.data(), b.data(), cReference.data(), size_n, size_k,
                          size_m);

  std::cout << "Running simulation...\n" << std::flush;
#ifdef MM_DYNAMIC_SIZES
  MatrixMultiplicationKernel(aKernel.data(), bKernel.data(), cKernel.data(),
                             size_n, size_k, size_m);
#else
  MatrixMultiplicationKernel(aKernel.data(), bKernel.data(), cKernel.data());
#endif
  std::cout << "Verifying results...\n" << std::flush;

  const auto cTest = Unpack<kMemoryWidthM>(cKernel);

  for (unsigned i = 0; i < size_n; ++i) {
    for (unsigned j = 0; j < size_m; ++j) {
      const auto testVal = make_signed<Data_t>(cTest[i * size_m + j]);
      const auto refVal = make_signed<Data_t>(cReference[i * size_m + j]);
      const Data_t diff = std::abs(testVal - refVal);
      bool mismatch;
      if (std::is_floating_point<Data_t>::value) {
        mismatch = diff / refVal > static_cast<Data_t>(1e-3);
      } else {
        mismatch = diff != 0;
      }
      if (mismatch) {
        std::cerr << "Mismatch at (" << i << ", " << j << "): " << testVal
                  << " vs. " << refVal << "\n";
        return 1;
      }
    }
  }
  std::cout << "Matrix-matrix multiplication successfully verified.\n";

  return 0;
}
