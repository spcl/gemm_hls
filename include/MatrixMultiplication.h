/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <type_traits>
#include "Config.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Resource.h"
#include "hlslib/xilinx/Stream.h"

using hlslib::Stream;

constexpr int kSeed = 5; // For initializing matrices for testing
constexpr unsigned kPipeDepth = 4;

// Memory bus in K-dimension
constexpr int kMemoryWidthK = kMemoryWidthBytesK / sizeof(Data_t);
static_assert(kMemoryWidthBytesK % sizeof(Data_t) == 0,
              "Memory width in K not divisable by size of data type.");
using MemoryPackK_t = hlslib::DataPack<Data_t, kMemoryWidthK>;

// Memory bus in M-dimension
constexpr int kMemoryWidthM = kMemoryWidthBytesM / sizeof(Data_t);
static_assert(kMemoryWidthBytesM % sizeof(Data_t) == 0,
              "Memory width in M not divisable by size of data type.");
using MemoryPackM_t = hlslib::DataPack<Data_t, kMemoryWidthM>;

// Internal compute buses
using ComputePackN_t = hlslib::DataPack<Data_t, kComputeTileSizeN>;
using ComputePackM_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;
using OutputPack_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;

#ifndef MM_TRANSPOSED_A

// On-chip transpose of A
constexpr int kTransposeWidth = kTransposeWidthBytes / sizeof(Data_t);
static_assert(kTransposeWidthBytes % sizeof(Data_t) == 0,
              "Transpose width must be divisable by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0,
              "Transpose width must be divisable by memory port width.");

using MemoryPackA_t = MemoryPackK_t;
constexpr decltype(kMemoryWidthK) kMemoryWidthA = kMemoryWidthK;

#else // MM_TRANSPOSED_A

// Memory bus in N-dimension (for transposed A)
constexpr int kMemoryWidthN = kMemoryWidthBytesN / sizeof(Data_t);
static_assert(kMemoryWidthBytesN % sizeof(Data_t) == 0,
              "Memory width in N not divisable by size of data type.");
using MemoryPackN_t = hlslib::DataPack<Data_t, kMemoryWidthN>;
using MemoryPackA_t = MemoryPackN_t;
constexpr decltype(kMemoryWidthN) kMemoryWidthA = kMemoryWidthN;

constexpr unsigned long kOuterTileSizeNMemory = kOuterTileSizeN / kMemoryWidthN;
static_assert(
    kOuterTileSizeN % kMemoryWidthN == 0,
    "Outer memory tile size in N must be divisable by memory port width.");

inline unsigned SizeNMemory(unsigned n) {
  #pragma HLS INLINE
  return n / kMemoryWidthN;
}

#endif // MM_TRANSPOSED_A

constexpr unsigned long kOuterTileSizeMMemory = kOuterTileSizeM / kMemoryWidthM;
static_assert(
    kOuterTileSizeM % kMemoryWidthM == 0,
    "Outer memory tile size in M must be divisable by memory port width.");

constexpr unsigned long kInnerTilesN = kOuterTileSizeN / kInnerTileSizeN;
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr unsigned long kInnerTilesM = kOuterTileSizeM / kComputeTileSizeM;
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0,
              "Outer tile size must be divisable by compute tile size in M.");

constexpr unsigned long kComputeTilesN = kInnerTileSizeN / kComputeTileSizeN;
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0,
              "Inner tile size must be divisable by compute tile size.");

#ifndef MM_DYNAMIC_SIZES

static_assert(kSizeK % kMemoryWidthK == 0,
              "K must be divisable by memory width.");

#ifndef MM_TRANSPOSED_A

static_assert(kSizeK % kTransposeWidth == 0,
              "K must be divisable by the transpose width.");

#endif

#endif

inline unsigned SizeKMemory(unsigned k) {
  #pragma HLS INLINE
  return k / kMemoryWidthK;
}

inline unsigned SizeMMemory(unsigned m) {
  #pragma HLS INLINE
  return m / kMemoryWidthM;
}

inline unsigned OuterTilesN(unsigned n) {
  #pragma HLS INLINE
  return (n + kOuterTileSizeN - 1) / kOuterTileSizeN;
}

inline unsigned OuterTilesM(unsigned m) {
  #pragma HLS INLINE
  return (m + kOuterTileSizeM - 1) / kOuterTileSizeM;
}

inline unsigned long TotalReadsFromA(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
  #pragma HLS INLINE
  return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
         kOuterTileSizeN * size_k;
}

inline unsigned long TotalReadsFromB(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
  #pragma HLS INLINE
  return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
         kOuterTileSizeM * size_k;
}

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

#ifdef MM_TRANSPOSED_A
void MatrixMultiplicationKernel(MemoryPackN_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#else
void MatrixMultiplicationKernel(MemoryPackK_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#endif
#ifdef MM_DYNAMIC_SIZES
                                ,
                                const unsigned size_n, const unsigned size_k,
                                const unsigned size_m
#endif
);

}
