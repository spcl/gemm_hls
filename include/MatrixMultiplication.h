/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <type_traits>
#include "Config.h"
#include "hlslib/DataPack.h"
#include "hlslib/Resource.h"

constexpr int kSeed = 5; // For initializing matrices for testing
constexpr int kFifoDepth = 8;

constexpr int kMemoryWidthK = kMemoryWidthBytesK / sizeof(Data_t);
static_assert(kMemoryWidthBytesK % sizeof(Data_t) == 0,
              "Memory width in K not divisable by size of data type.");
using MemoryPackK_t = hlslib::DataPack<Data_t, kMemoryWidthK>;
constexpr int kMemoryWidthM = kMemoryWidthBytesM / sizeof(Data_t);
static_assert(kMemoryWidthBytesM % sizeof(Data_t) == 0,
              "Memory width in M not divisable by size of data type.");
using MemoryPackM_t = hlslib::DataPack<Data_t, kMemoryWidthM>;
using ComputePackN_t = hlslib::DataPack<Data_t, kComputeTileSizeN>;
using ComputePackM_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;
using OutputPack_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;

constexpr int kTransposeWidth = kTransposeWidthBytes / sizeof(Data_t);
static_assert(kTransposeWidthBytes % sizeof(Data_t) == 0,
              "Transpose width must be divisable by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0,
              "Transpose width must be divisable by memory port width.");

constexpr int kSizeKMemory = kSizeK / kMemoryWidthK;
static_assert(kSizeK % kMemoryWidthK == 0,
              "K must be divisable by memory width.");

constexpr int kSizeMMemory = kSizeM / kMemoryWidthM;
static_assert(kSizeM % kMemoryWidthM == 0,
              "M must be divisable by memory width.");

constexpr int kOuterTileSizeMMemory = kOuterTileSizeM / kMemoryWidthM;
static_assert(
    kOuterTileSizeM % kMemoryWidthM == 0,
    "Outer memory tile size in M must be divisable by memory port width.");

constexpr int kOuterTilesN = kSizeN / kOuterTileSizeN;
static_assert(kSizeN % kOuterTileSizeN == 0,
              "N must be divisable by the outer tile size in N.");

constexpr int kOuterTilesM = kSizeM / kOuterTileSizeM;
static_assert(kSizeM % kOuterTileSizeM == 0,
              "M must be divisable by the outer tile size in M.");

constexpr int kInnerTilesN = kOuterTileSizeN / kInnerTileSizeN;
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr int kInnerTilesM = kOuterTileSizeM / kComputeTileSizeM;
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0,
              "Outer tile size must be divisable by compute tile size in M.");

constexpr int kComputeTilesN = kInnerTileSizeN / kComputeTileSizeN;
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0,
              "Inner tile size must be divisable by compute tile size.");

template <typename T,
          class = typename std::enable_if<std::is_integral<T>::value, T>::type>
constexpr T PowerOfTwo(T number, unsigned char power) {
  return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

#ifdef MM_ADD_RESOURCE
#define MM_ADD_RESOURCE_PRAGMA(var)                                 \
  HLSLIB_RESOURCE_PRAGMA(var, MM_ADD_RESOURCE)
#else
#define MM_ADD_RESOURCE_PRAGMA(var)
#endif

#ifdef MM_MULT_RESOURCE
#define MM_MULT_RESOURCE_PRAGMA(var)                                 \
  HLSLIB_RESOURCE_PRAGMA(var, MM_MULT_RESOURCE)
#else
#define MM_MULT_RESOURCE_PRAGMA(var)
#endif

extern "C" {

void MatrixMultiplicationKernel(MemoryPackK_t const *aMem,
                                MemoryPackM_t const *bMem, MemoryPackM_t *cMem);
}
