/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Utility.h"
#include "MatrixMatrix.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>
#ifdef MM_HAS_BLAS
#include "cblas.h"
#endif

// Fallback
template <typename T, class OperatorMap, class OperatorReduce>
void CallBLAS(T const *a, T const *b, T *c) {
  Naive<OperatorMap, OperatorReduce>(a, b, c, kSizeN, kSizeM, kSizeP);
}

#ifdef MM_HAS_BLAS
template <>
void CallBLAS<float, hlslib::op::Multiply<float>, hlslib::op::Add<float>>(
    float const *a, float const *b, float *c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSizeN, kSizeP, kSizeM,
              1.0, a, kSizeM, b, kSizeP, 0.0, c, kSizeP);
}
template <>
void CallBLAS<double, hlslib::op::Multiply<double>, hlslib::op::Add<double>>(
    double const *a, double const *b, double *c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSizeN, kSizeP, kSizeM,
              1.0, a, kSizeM, b, kSizeP, 0.0, c, kSizeP);
}
#endif

int main() {

  std::vector<Data_t> a(kSizeN * kSizeM);
  std::vector<Data_t> b(kSizeM * kSizeP);
  std::vector<Data_t> cReference(kSizeN * kSizeP, 0);

  std::default_random_engine rng(kSeed);
  typename std::conditional<
      std::is_integral<Data_t>::value, std::uniform_int_distribution<unsigned long>,
      std::uniform_real_distribution<double>>::type dist(1, 10);

  std::for_each(a.begin(), a.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
  std::for_each(b.begin(), b.end(),
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });

  const auto aKernel = Pack(a);
  const auto bKernel = Pack(b);
  auto cKernel = Pack(cReference);

  std::stringstream ss;
  ss << kGoldenDir << "gemm_n" << kSizeN << "_m" << kSizeM << "_p" << kSizeP
     << "_s" << kSeed << ".dat";
  const auto goldenFileName = ss.str();
  std::ifstream goldenFile(goldenFileName,
                           std::ios_base::in | std::ios_base::binary);
  if (goldenFile.good()) {
    std::cout << "Using cached golden result." << std::endl;
    goldenFile.read(reinterpret_cast<char *>(&cReference[0]),
                    cReference.size() * sizeof(Data_t));
  } else {
    std::cout << "No cached result found. Running naive implementation..."
              << std::flush;
    CallBLAS<Data_t, OperatorMap, OperatorReduce>(a.data(), b.data(),
                                                  cReference.data());
    std::cout << " Done.\n";
    std::ofstream goldenFileOut(goldenFileName,
                                std::ios_base::out | std::ios_base::binary);
    if (!goldenFileOut.good()) {
      std::cerr << "Failed to open output file \"" << goldenFileName
                << "\". Cannot cache result." << std::endl;
    } else {
      goldenFileOut.write(reinterpret_cast<char *>(&cReference[0]),
                          cReference.size() * sizeof(Data_t));
    }
  }

  std::cout << "Running hardware emulation..." << std::flush;
  MatrixMatrix(aKernel.data(), bKernel.data(), cKernel.data());
  std::cout << " Done.\n";

  const auto cTest = Unpack(cKernel);

  std::cout << "Verifying results..." << std::endl;
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
