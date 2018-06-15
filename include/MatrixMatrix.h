/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
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
using MemoryPack_t = hlslib::DataPack<Data_t, kMemoryWidth>;

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

constexpr int kOuterTilesN = kSizeN / kOuterTileSize;
static_assert(kSizeN % kOuterTileSize == 0,
              "N must be divisable by the outer tile size.");

constexpr int kOuterTilesM = kSizeM / kOuterTileSize;
static_assert(kSizeM % kOuterTileSize == 0,
              "M must be divisable by the outer tile size.");

constexpr int kOuterTilesP = kSizeP / kOuterTileSize;
static_assert(kSizeP % kOuterTileSize == 0,
              "P must be divisable by the outer tile size.");

constexpr int kInnerTiles = kOuterTileSize / kInnerTileSize;
static_assert(kInnerTileSize % kInnerTileSize == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr int kInnerTileSizeMemory = kInnerTileSize / kMemoryWidth;
static_assert(kInnerTileSize % kMemoryWidth == 0,
              "Inner tile size must be divisable by memory width");

constexpr int kOuterTileSizeMemory = kOuterTileSize / kMemoryWidth;
static_assert(kOuterTileSize % kMemoryWidth == 0,
              "Outer tile size must be divisable by memory width");

template <typename T,
          class = typename std::enable_if<std::is_integral<T>::value, T>::type>
constexpr T PowerOfTwo(T number, unsigned char power) {
  return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

// Force HLS to implement these with BRAM by setting the minimum depth to 128
constexpr int kTransposeDepth = PowerOfTwo<int>(4 * kInnerTileSize, 0);

extern "C" {

void MatrixMatrix(Data_t const *aMem, MemoryPack_t const *bMem, Data_t *cMem);

}
