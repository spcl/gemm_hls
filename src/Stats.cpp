#include "MatrixMultiplication.h"
#include "Memory.h"

// Prints the expected performance of the current configuration in hardware.
// If a different frequency is achieved, it can be passed as the first argument
// to the executable.
int main(int argc, char **argv) {
  float frequency = kFrequency;
  if (argc > 1) {
    frequency = std::stof(argv[1]);
  }
  const unsigned long long nOps =
      2 * static_cast<unsigned long long>(kSizeN) * kSizeK * kSizeM;
  std::cout << "Frequency:            " << frequency << " MHz\n";
  std::cout << "Number of operations: " << nOps << " ("
            << static_cast<float>(nOps) << ")\n";
  const auto expected_runtime =
      static_cast<float>(kOuterTilesN) * kOuterTilesM *
      (kSizeK * kInnerTilesN * kInnerTilesM +
       kInnerTilesN * (kComputeTileSizeM * kInnerTilesM +
                       kComputeTilesN * kComputeTileSizeN * kInnerTilesM)) /
      (1e6 * frequency);
  const auto expected_perf = 1e-9 * nOps / expected_runtime;
  const auto peak_perf = 2e-3 * kInnerTileSizeN * kComputeTileSizeM * frequency;
  std::cout << "Expected runtime:     " << expected_runtime << " seconds\n";
  std::cout << "Peak runtime:         " << nOps / (1e9 * peak_perf)
            << " seconds\n";
  std::cout << "Expected performance: " << expected_perf << " GOp/s\n";
  std::cout << "Peak performance:     " << peak_perf << " GOp/s\n";
  std::cout << "Tiles in N (outer/inner): " << kOuterTilesN << " / "
            << kInnerTilesN << "\n";
  std::cout << "Tiles in M (outer/inner): " << kOuterTilesM << " / "
            << kInnerTilesM << "\n";
  return 0;
}
