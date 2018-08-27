/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "MatrixMultiplication.h"
#include "Utility.h"
#include "hlslib/SDAccel.h"

int main(int argc, char **argv) {
  // Use fixed seed to enable comparison to saved golden values
  std::default_random_engine rng(kSeed);
  typename std::conditional<std::is_integral<Data_t>::value,
                            std::uniform_int_distribution<unsigned long>,
                            std::uniform_real_distribution<double>>::type
      dist(1, 10);

  bool verify = true;
  if (argc > 1) {
    const std::string verifyArg(argv[1]);
    if (verifyArg == "off") {
      verify = false;
    } else if (verifyArg != "on") {
      std::cerr << "Argument should be [on/off]" << std::endl;
      return 1;
    }
  }

  std::vector<Data_t> a, b, cRef;
  std::vector<MemoryPack_t> aMem, bMem, cMem;
  std::cout << "Initializing host memory..." << std::flush;
  if (verify) {
    a = std::vector<Data_t>(kSizeN * kSizeK);
    std::for_each(a.begin(), a.end(),
                  [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
    b = std::vector<Data_t>(kSizeK * kSizeM);
    std::for_each(b.begin(), b.end(),
                  [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
    cRef = std::vector<Data_t>(kSizeN * kSizeM, 0);

    aMem = Pack(a);
    bMem = Pack(b);
    cMem = Pack(cRef);
  }
  std::cout << " Done.\n";

  try {
    std::cout << "Initializing OpenCL context..." << std::flush;
    hlslib::ocl::Context context;
    std::cout << " Done.\n";

    std::cout << "Initializing device memory..." << std::flush;
    auto aDevice = context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::read>(
        hlslib::ocl::MemoryBank::bank0, &aMem[0], &aMem[kSizeN * kSizeKMemory]);
    auto bDevice = context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::read>(
        hlslib::ocl::MemoryBank::bank1, &bMem[0], &bMem[kSizeK * kSizeMMemory]);
    auto cDevice = context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::write>(
        hlslib::ocl::MemoryBank::bank1, &cMem[0], &cMem[kSizeN * kSizeMMemory]);
    std::cout << " Done.\n";

    std::cout << "Programming device..." << std::flush;
    auto program = context.MakeProgram("MatrixMultiplication.xclbin");
    auto kernel = program.MakeKernel("MatrixMultiplicationKernel", aDevice,
                                     bDevice, cDevice);
    std::cout << " Done.\n";

    std::cout << "Executing kernel..." << std::flush;
    const auto elapsed = kernel.ExecuteTask();
    std::cout << " Done.\n";

    const auto perf = 1e-9 *
                      (2 * static_cast<float>(kSizeN) * kSizeK * kSizeM) /
                      elapsed.first;

    std::cout << "Kernel executed in " << elapsed.first
              << " seconds, corresponding to a performance of " << perf
              << " GOp/s.\n";

    if (verify) {
      std::cout << "Copying back result..." << std::flush;
      cDevice.CopyToHost(cMem.begin());
      std::cout << " Done.\n";
    }

  } catch (std::runtime_error const &err) {
    std::cerr << "Execution failed with error: \"" << err.what() << "\"."
              << std::endl;
    return 1;
  }

  // Run reference implementation
  if (verify) {
    ReferenceImplementation(a.data(), b.data(), cRef.data());

    // Convert to single element vector
    const auto cTest = Unpack(cMem);

    for (int i = 0; i < kSizeN; ++i) {
      for (int j = 0; j < kSizeM; ++j) {
        const auto testVal = cTest[i * kSizeM + j];
        const auto refVal = cRef[i * kSizeM + j];
        const auto diff = std::abs(testVal - refVal);
        if (diff > static_cast<Data_t>(1e-3)) {
          std::cerr << "Mismatch at (" << i << ", " << j << "): " << testVal
                    << " vs. " << refVal << "\n";
          return 1;
        }
      }
    }
    std::cout << "Successfully verified." << std::endl;
  }

  return 0;
}
