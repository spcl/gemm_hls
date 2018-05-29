/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "Config.h"
#include "hlslib/DataPack.h"
#include <type_traits>

constexpr int kSeed = 5; // For initializing matrices for testing

constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);
static_assert(kMemoryWidthBytes % sizeof(Data_t) == 0,
              "Memory width not divisable by size of data type.");
constexpr int kKernelPerMemory = kMemoryWidth / kKernelWidth;
static_assert(kMemoryWidth % kKernelWidth == 0,
              "Memory width must be divisable by kernel width.");
using KernelPack_t = hlslib::DataPack<Data_t, kKernelWidth>;
using MemoryPack_t = hlslib::DataPack<KernelPack_t, kKernelPerMemory>;

// constexpr int kSizeKernel = kSize / kKernelWidth;
// static_assert(kSize % kKernelWidth == 0,
//               "Matrix dimensions must be divisable by kernel width.");

// constexpr int kSizeMemory = kSize / kMemoryWidth;
// static_assert(kSize % kMemoryWidth == 0,
//               "Matrix dimensions must be divisable by memory width.");

constexpr int kSizeMMemory = kSizeM / kMemoryWidth;
static_assert(kSizeM % kMemoryWidth == 0,
              "M must be divisable by memory width.");

constexpr int kSizePMemory = kSizeP / kMemoryWidth;
static_assert(kSizeP % kMemoryWidth == 0,
              "P must be divisable by memory width.");

constexpr int kBlocksN = kSizeN / kTileSizeN;
static_assert(kSizeN % kTileSizeN == 0,
              "N must be divisable by tile size in N.");

constexpr int kBlocksP = kSizeP / kTileSizeP;
static_assert(kSizeP % kTileSizeP == 0,
              "P must be divisable by tile size in P.");

constexpr int kTileSizePKernel = kTileSizeP / kKernelWidth;
static_assert(kTileSizeP % kKernelWidth == 0,
              "Tile size in P must be divisable by kernel width");

constexpr int kTileSizePMemory = kTileSizeP / kMemoryWidth;
static_assert(kTileSizeP % kMemoryWidth == 0,
              "Tile size in P must be divisable by memory width");

static_assert(kTileSizePKernel >= kTileSizeN, "Horizontal tile size with "
                                              "vectorization must be higher "
                                              "than vertical tile size.");

template <typename T,
          class = typename std::enable_if<std::is_integral<T>::value, T>::type>
constexpr T PowerOfTwo(T number, unsigned char power) {
  return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

// Force HLS to implement these with BRAM by setting the minimum depth to 128
constexpr int kTransposeDepth = PowerOfTwo<int>(4 * kTileSizeN, 0);

extern "C" {

void MatrixMatrix(MemoryPack_t const *aMem, MemoryPack_t const *bMem,
                  MemoryPack_t *cMem);
}
