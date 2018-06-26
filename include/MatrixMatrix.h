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

constexpr int kSizeKMemory = kSizeK / kMemoryWidth;
static_assert(kSizeK % kMemoryWidth == 0,
              "K must be divisable by memory width.");

constexpr int kSizeMMemory = kSizeM / kMemoryWidth;
static_assert(kSizeM % kMemoryWidth == 0,
              "M must be divisable by memory width.");

constexpr int kOuterTilesN = kSizeN / kOuterTileSize;
static_assert(kSizeN % kOuterTileSize == 0,
              "N must be divisable by the outer tile size.");

constexpr int kOuterTilesK = kSizeK / kOuterTileSize;
static_assert(kSizeK % kOuterTileSize == 0,
              "K must be divisable by the outer tile size.");

constexpr int kOuterTilesM = kSizeM / kOuterTileSize;
static_assert(kSizeM % kOuterTileSize == 0,
              "M must be divisable by the outer tile size.");

constexpr int kInnerTilesN = kOuterTileSize / kInnerTileSizeN;
static_assert(kOuterTileSize % kInnerTileSizeN == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr int kInnerTilesM = kOuterTileSize / kInnerTileSizeM;
static_assert(kOuterTileSize % kInnerTileSizeM == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr int kInnerTileSizeMMemory = kInnerTileSizeM / kMemoryWidth;
static_assert(kInnerTileSizeM % kMemoryWidth == 0,
              "Inner tile size must be divisable by memory width");

constexpr int kOuterTileSizeMemory = kOuterTileSize / kMemoryWidth;
static_assert(kOuterTileSize % kMemoryWidth == 0,
              "Outer tile size must be divisable by memory width");

static_assert(kInnerTileSizeMMemory > 1,
              "Vectorized inner tile size must be larger than 1, "
              "otherwise HLS will not pipeline");

template <typename T,
          class = typename std::enable_if<std::is_integral<T>::value, T>::type>
constexpr T PowerOfTwo(T number, unsigned char power) {
  return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

extern "C" {

void MatrixMatrix(Data_t const *aMem, MemoryPack_t const *bMem,
                  MemoryPack_t *cMem);
}
