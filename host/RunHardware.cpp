/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "MatrixMultiplication.h"
#include "Utility.h"
#include "hlslib/xilinx/OpenCL.h"
#include "hlslib/xilinx/Utility.h"
#ifdef MM_POWER_METER
#include "PowerMeter.h"
#endif

void PrintUsage() {
#ifndef MM_DYNAMIC_SIZES
  std::cerr
      << "Usage: ./RunHardware.exe <mode [hw/hw_emu]> [<verify [on/off]>]\n"
      << std::flush;
#else
  std::cerr << "Usage: ./RunHardware.exe N K M [<mode [hw/hw_emu]>] [<verify "
               "[on/off]>]\n"
            << std::flush;
#endif
}

int main(int argc, char **argv) {
  std::default_random_engine rng(kSeed);
  typename std::conditional<std::is_integral<Data_t>::value,
                            std::uniform_int_distribution<unsigned long>,
                            std::uniform_real_distribution<double>>::type
      dist(1, 10);

  bool emulation = false;
  bool verify = true;
  hlslib::UnsetEnvironmentVariable("XCL_EMULATION_MODE");
  std::string path = "MatrixMultiplication_hw.xclbin";
#ifdef MM_DYNAMIC_SIZES
  if (argc > 6 || argc < 4) {
    PrintUsage();
    return 1;
  }
  const unsigned size_n = std::stoul(argv[1]);
  const unsigned size_k = std::stoul(argv[2]);
  const unsigned size_m = std::stoul(argv[3]);
  int next_arg = 4;
  if (size_k % kMemoryWidthK != 0) {
    std::cerr << "K (" << size_k
              << ") must be divisable by the memory width in K ("
              << kMemoryWidthK << ")." << std::endl;
    return 1;
  }
  if (size_m % kMemoryWidthM != 0) {
    std::cerr << "M (" << size_m
              << ") must be divisable by the memory width in M ("
              << kMemoryWidthM << ")." << std::endl;
    return 1;
  }
#else
  if (argc > 3) {
    PrintUsage();
    return 1;
  }
  constexpr auto size_n = kSizeN;
  constexpr auto size_k = kSizeK;
  constexpr auto size_m = kSizeM;
  int next_arg = 1;
#endif
  if (next_arg < argc) {
    const std::string emulation_arg(argv[next_arg++]);
    if (emulation_arg == "hw_emu") {
      emulation = true;
      hlslib::SetEnvironmentVariable("XCL_EMULATION_MODE", "hw_emu");
      path = "MatrixMultiplication_hw_emu.xclbin";
    } else if (emulation_arg != "hw") {
      PrintUsage();
      return 1;
    }
  }
  if (next_arg < argc) {
    const std::string verify_arg(argv[next_arg++]);
    if (verify_arg == "off") {
      verify = false;
    } else if (verify_arg != "on") {
      PrintUsage();
      return 1;
    }
  }

  std::vector<Data_t> a, b, cRef;
  std::vector<MemoryPackK_t, hlslib::ocl::AlignedAllocator<MemoryPackK_t, 4096>>
      aMem;
  std::vector<MemoryPackM_t, hlslib::ocl::AlignedAllocator<MemoryPackM_t, 4096>>
      bMem, cMem;
  std::cout << "Initializing host memory..." << std::flush;
  if (verify) {
    a = decltype(a)(size_n * size_k);
    std::for_each(a.begin(), a.end(),
                  [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
    b = decltype(b)(size_k * size_m);
    std::for_each(b.begin(), b.end(),
                  [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });
    cRef = decltype(cRef)(size_n * size_m, 0);

    aMem = Pack<kMemoryWidthK>(a);
    bMem = Pack<kMemoryWidthM>(b);
    cMem = Pack<kMemoryWidthM>(cRef);
  }
  std::cout << " Done.\n";

  try {
    std::cout << "Initializing OpenCL context...\n" << std::flush;
    hlslib::ocl::Context context;

    std::cout << "Programming device...\n" << std::flush;
    auto program = context.MakeProgram(path);

    std::cout << "Initializing device memory...\n" << std::flush;
#ifdef MM_TWO_DIMMS
    auto aDevice = context.MakeBuffer<MemoryPackK_t, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 0, size_n * size_k / kMemoryWidthK);
    auto bDevice = context.MakeBuffer<MemoryPackM_t, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 1, size_k * size_m / kMemoryWidthM);
    auto cDevice =
        context.MakeBuffer<MemoryPackM_t, hlslib::ocl::Access::write>(
            hlslib::ocl::StorageType::DDR, 1, size_n * size_m / kMemoryWidthM);
#else
    auto aDevice = context.MakeBuffer<MemoryPackK_t, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 1, size_n * size_k / kMemoryWidthK);
    auto bDevice = context.MakeBuffer<MemoryPackM_t, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 1, size_k * size_m / kMemoryWidthM);
    auto cDevice =
        context.MakeBuffer<MemoryPackM_t, hlslib::ocl::Access::write>(
            hlslib::ocl::StorageType::DDR, 1, size_n * size_m / kMemoryWidthM);
#endif

    if (verify) {
      std::cout << "Copying memory to device...\n" << std::flush;
      aDevice.CopyFromHost(aMem.cbegin());
      bDevice.CopyFromHost(bMem.cbegin());
      cDevice.CopyFromHost(cMem.cbegin());
    }

    std::cout << "Creating kernel...\n" << std::flush;
#ifndef MM_DYNAMIC_SIZES
    auto kernel = program.MakeKernel("MatrixMultiplicationKernel", aDevice,
                                     bDevice, cDevice);
#else
    auto kernel = program.MakeKernel("MatrixMultiplicationKernel", aDevice,
                                     bDevice, cDevice, size_n, size_k, size_m);
#endif

#ifdef MM_POWER_METER
    PowerMeter pm(10);  // 10 ms sampling rate
    pm.Start();
#endif

    std::cout << "Executing kernel...\n" << std::flush;
    const auto elapsed = kernel.ExecuteTask();

#ifdef MM_POWER_METER
    pm.Stop();
    const auto power = pm.GetSamples();
    float average_power = 0;
    for (auto &i : power) {
      average_power += std::get<1>(i);
    }
    average_power /= power.size();
#endif

    const auto perf = 1e-9 *
                      (2 * static_cast<float>(size_n) * size_k * size_m) /
                      elapsed.first;

    std::cout << "Kernel executed in " << elapsed.first
              << " seconds, corresponding to a performance of " << perf
              << " GOp/s.\n";

#ifdef MM_POWER_METER
    std::cout << "Measured an average power of " << average_power
              << " W for the full system.\n";
#endif

    if (verify) {
      std::cout << "Copying back result...\n" << std::flush;
      cDevice.CopyToHost(cMem.begin());
    }

  } catch (std::runtime_error const &err) {
    std::cerr << "Execution failed with error: \"" << err.what() << "\"."
              << std::endl;
    return 1;
  }

  // Run reference implementation
  if (verify) {
    std::cout << "Running reference implementation...\n" << std::flush;
    ReferenceImplementation(a.data(), b.data(), cRef.data(), size_n, size_k,
                            size_m);

    std::cout << "Verifying result...\n" << std::flush;
    // Convert to single element vector
    const auto cTest = Unpack<kMemoryWidthM>(cMem);

    for (size_t i = 0; i < size_n; ++i) {
      for (size_t j = 0; j < size_m; ++j) {
        const auto testVal = make_signed<Data_t>(cTest[i * size_m + j]);
        const auto refVal = make_signed<Data_t>(cRef[i * size_m + j]);
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
    std::cout << "Successfully verified." << std::endl;
  }

  return 0;
}
