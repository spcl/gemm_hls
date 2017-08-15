/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "Config.h"
#include "hlslib/DataPack.h"

constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);
static_assert(kMemoryWidthBytes % sizeof(Data_t) == 0,
              "Memory width not divisable by size of data type.");
constexpr int kKernelPerMemory = kMemoryWidth / kKernelWidth;
static_assert(kMemoryWidth % kKernelWidth == 0,
              "Memory width must be divisable by kernel width.");
using KernelPack_t = hlslib::DataPack<Data_t, kKernelWidth>;
using MemoryPack_t = hlslib::DataPack<Data_t, kMemoryWidth>;

constexpr int kSizeKernel = kSize / kKernelWidth;
static_assert(kSize % kKernelWidth == 0,
              "Matrix dimensions must be divisable by kernel width.");

// constexpr int kSizeMemory = kSize / kMemoryWidth;
// static_assert(kSize % kMemoryWidth == 0,
//               "Matrix dimensions must be divisable by memory width.");

constexpr int kBlocksN = kSize / kTileSizeN;
static_assert(kSize % kTileSizeN == 0,
              "N must be divisable by tile size in N.");

constexpr int kBlocksP = kSize / kTileSizeP;
static_assert(kSize % kTileSizeP == 0,
              "P must be divisable by tile size in P.");

constexpr int kTileSizePKernel = kTileSizeP / kKernelWidth;
static_assert(kTileSizePKernel % kKernelWidth == 0,
              "Tile size in P must be divisable by kernel width");

// constexpr int kTileSizePMemory = kTileSizeP / kMemoryWidth;
// static_assert(kTileSizePMemory % kMemoryWidth == 0,
//               "Tile size in P must be divisable by memory width");

extern "C" {

void MatrixMatrix(Data_t const *aMem, KernelPack_t const *bMem,
                  KernelPack_t *cMem);
}
