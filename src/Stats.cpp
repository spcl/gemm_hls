#include "MatrixMatrix.h"

int main(int argc, char **argv) {
  float frequency = kFrequency;
  if (argc > 1) {
    frequency = std::stof(argv[1]);
  }
  const unsigned long long nOps =
      2 * static_cast<unsigned long long>(kSize) * kSize * kSize;
  std::cout << "Frequency:            " << frequency << " MHz\n";
  std::cout << "Number of operations: " << nOps << " ("
            << static_cast<float>(nOps) << ")\n";
  const auto peakPerf = 2e-3 * kTileSizeN * kKernelWidth * frequency;
  std::cout << "Peak performance:     " << peakPerf << " GOp/s\n";
  std::cout << "Peak runtime:         " << nOps / (1e9 * peakPerf)
            << " seconds.\n";
  std::cout << "Transpose FIFO depth: " << kTransposeDepth << "\n";
  return 0;
}
