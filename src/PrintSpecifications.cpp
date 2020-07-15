#include "MatrixMultiplication.h"
#include "Memory.h"

void PrintUsage(char **argv) {
#ifndef MM_DYNAMIC_SIZES
  std::cerr << "Usage: " << argv[0] << " [<routed frequency>]\n" << std::flush;
#else
  std::cerr << "Usage: " << argv[0] << " N K M [<routed_frequency>]\n"
            << std::flush;
#endif
}

// Prints the expected performance of the current configuration in hardware.
// If a different frequency is achieved, it can be passed as the first argument
// to the executable.
int main(int argc, char **argv) {
#ifdef MM_DYNAMIC_SIZES
  if (argc > 5 || argc < 4) {
    PrintUsage(argv);
    return 1;
  }
  const unsigned size_n = std::stoul(argv[1]);
  const unsigned size_k = std::stoul(argv[2]);
  const unsigned size_m = std::stoul(argv[3]);
  int next_arg = 4;
#else
  if (argc > 2) {
    PrintUsage(argv);
    return 1;
  }
  constexpr auto size_n = kSizeN;
  constexpr auto size_k = kSizeK;
  constexpr auto size_m = kSizeM;
  int next_arg = 1;
#endif
  float frequency = kFrequency;
  if (argc > next_arg) {
    frequency = std::stof(argv[next_arg]);
  }
  const unsigned long long nOps =
      2 * static_cast<unsigned long long>(size_n) * size_k * size_m;
  std::cout << "Frequency:            " << frequency << " MHz\n";
  std::cout << "Number of operations: " << nOps << " ("
            << static_cast<float>(nOps) << ")\n";
  const auto expected_runtime =
      static_cast<float>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
      (size_k * kInnerTilesN * kInnerTilesM +
       kInnerTilesN * (kComputeTileSizeM * kInnerTilesM +
                       kComputeTilesN * kComputeTileSizeN * kInnerTilesM)) /
      (1e6 * frequency);
  const auto ideal_runtime = (static_cast<float>(size_n) * size_k * size_m /
                              (kInnerTileSizeN * kComputeTileSizeM)) /
                             (1e6 * frequency);
  const auto expected_perf = 1e-9 * nOps / expected_runtime;
  const auto ideal_perf =
      2e-3 * kInnerTileSizeN * kComputeTileSizeM * frequency;
  std::cout << "Expected runtime:     " << expected_runtime << " seconds\n";
  std::cout << "Ideal runtime:        " << ideal_runtime << " seconds\n";
  std::cout << "Percentage of deal:   "
            << 100 * ideal_runtime / expected_runtime << "%\n";
  std::cout << "Expected performance: " << expected_perf << " GOp/s\n";
  std::cout << "Ideal performance:    " << ideal_perf << " GOp/s\n";
  std::cout << "Compute tiles: " << kInnerTileSizeN << "x" << kComputeTileSizeM
            << " (" << kInnerTileSizeN * kComputeTileSizeM
            << " parallel adders/multipliers)\n";
  std::cout << "Memory tile size: " << kOuterTileSizeN << "x" << kOuterTileSizeM
            << "\n";
  std::cout << "Tiles in N (outer/inner): " << OuterTilesN(size_n) << " / "
            << kInnerTilesN << "\n";
  std::cout << "Tiles in M (outer/inner): " << OuterTilesM(size_m) << " / "
            << kInnerTilesM << "\n";
  const unsigned long long communicationVolume =
      size_n * size_m *
      (1 + size_k / kOuterTileSizeN + size_k / kOuterTileSizeM);
  std::cout << "Communication volume: " << communicationVolume << "\n";
  const double ioAccesses =
      communicationVolume / (3 * static_cast<double>(size_n) * size_m * size_k);
  std::cout << "I/O access fraction: " << ioAccesses << "\n";
  return 0;
}
