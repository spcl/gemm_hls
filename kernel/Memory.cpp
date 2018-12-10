/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Memory.h"
#include <cassert>

using hlslib::Stream;

int IndexA(int n0, int n1, int n2, int k0, int k1) {
  #pragma HLS INLINE
  const auto index =
      (n0 * kOuterTileSizeN + n1 * kInnerTileSizeN + n2) * kSizeKMemory +
      (k0 * (kTransposeWidth / kMemoryWidthK) + k1);
  assert(index < kSizeN * kSizeKMemory);
  return index;
}

int IndexB(int k, int m0, int m1m) {
  #pragma HLS INLINE
  const auto index = k * kSizeMMemory + (m0 * kOuterTileSizeMMemory + m1m);
  assert(index < kSizeK * kSizeMMemory);
  return index;
}

int IndexC(int n0, int n1, int m0, int m1m) {
  #pragma HLS INLINE
  const auto index = (n0 * kOuterTileSizeN + n1) * kSizeMMemory +
                     (m0 * kOuterTileSizeMMemory + m1m);
  assert(index < kSizeN * kSizeMMemory);
  return index;
}

void _ReadAInner(MemoryPackK_t const a[],
                 Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                 int n0, int n1, int n2, int k0, int k1) {

  #pragma HLS INLINE
  auto pack = a[IndexA(n0, n1, n2, k0, k1)];
ReadA_Unroll:
  for (int w = 0; w < kMemoryWidthK; ++w) {
    #pragma HLS UNROLL
    aSplit[k1 * kMemoryWidthK + w].Push(pack[w]); 
  }
}

template <int innerReads>
void _ReadAInnerLoop(
    MemoryPackK_t const a[],
    Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],

    int n0, int n1, int n2, int k0) {
#pragma HLS INLINE
ReadA_TransposeWidth:
  for (int k1 = 0; k1 < (kTransposeWidth / kMemoryWidthK); ++k1) { 
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    _ReadAInner(a, aSplit, n0, n1, n2, k0, k1);
  }
}

// Need a special case for kMemoryWidthK == kTransposeWidth, as Vivado HLS
// otherwise doesn't pipeline the loops (because the inner trip count is 1).
template <>
void _ReadAInnerLoop<1>(
    MemoryPackK_t const a[],
    Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth], int n0, int n1,
    int n2, int k0) {
  #pragma HLS INLINE
  #pragma HLS PIPELINE II=1
  #pragma HLS LOOP_FLATTEN
  _ReadAInner(a, aSplit, n0, n1, n2, k0, 0);
}

void ReadA(MemoryPackK_t const a[],
           Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth]) {

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 (kSizeK / kTransposeWidth) * kInnerTilesN * kInnerTileSizeN *
                 (kTransposeWidth / kMemoryWidthK) * MemoryPackK_t::kWidth) ==
                    kTotalReadsFromA,
                "Sanity check failed for ReadA");

ReadA_N0:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  ReadA_M0:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    ReadA_K0:
      for (int k0 = 0; k0 < kSizeK / kTransposeWidth; ++k0) {
      ReadA_N1:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        ReadA_N2:
          for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
            _ReadAInnerLoop<kTransposeWidth / kMemoryWidthK>(a, aSplit, n0, n1,
                                                            n2, k0);
          }
        }
      }
    }
  }
}

template <unsigned inner_tiles>
void _TransposeAInner(
    Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
    Stream<ComputePackN_t, kPipeDepth> &toKernel, const unsigned k) {
  #pragma HLS INLINE
  for (int n1 = 0; n1 < kOuterTileSizeN / kComputeTileSizeN; ++n1) {
    ComputePackN_t pack;
  TransposeA_N2:
    for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      pack[n2] = aSplit[k % kTransposeWidth].Pop();
      // Pop from each stream kOuterTileSizeN times in a row
      if (n2 == kComputeTileSizeN - 1) {
        toKernel.Push(pack);
      }
    }
  }
}

template <>
void _TransposeAInner<1>(
    Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
    Stream<ComputePackN_t, kPipeDepth> &toKernel, const unsigned k) {
  #pragma HLS INLINE
  for (int n1 = 0; n1 < kOuterTileSizeN; ++n1) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    ComputePackN_t pack;
    pack[0] = aSplit[k % kTransposeWidth].Pop();
    toKernel.Push(pack);
  }
}

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
void TransposeA(Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<ComputePackN_t, kPipeDepth> &toKernel) {
                
  static_assert(
      kOuterTileSizeN <= kInnerTilesN * kInnerTilesM,
      "In-memory transposition for A cannot keep up with the instantiated "
      "number of compute units. The number of inner tiles must be larger than "
      "the outer tile size in N.");

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 kSizeK * kOuterTileSizeN) == kTotalReadsFromA,
                "Sanity check failed for TransposeA");

TransposeA_N0:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  TransposeA_M0:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    TransposeA_K:
      for (int k = 0; k < kSizeK; ++k) {
        _TransposeAInner<kComputeTileSizeN>(aSplit, toKernel, k);
      }
    }
  }
}

#ifdef MM_CONVERT_A
void ConvertWidthA(Stream<Data_t, kPipeDepth> &narrow,
                   Stream<ComputePackN_t, kPipeDepth> &wide) {
ConvertWidthA_Outer:
  for (int i = 0; i < kTotalReadsFromA / ComputePackN_t::kWidth; ++i) {
    ComputePackN_t pack;
  ConvertWidthA_Compute:
    for (int w = 0; w < ComputePackN_t::kWidth; ++w) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      pack[w] = narrow.Pop();
    }
    wide.Push(pack);
  }
}
#endif

void ReadB(MemoryPackM_t const memory[],
           Stream<MemoryPackM_t, 2 * kOuterTileSizeM> &pipe) {
  static_assert(
      (static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kSizeK *
       kOuterTileSizeMMemory * MemoryPackM_t::kWidth) == kTotalReadsFromB,
      "Sanity check failed for ReadB");

ReadB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  ReadB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    ReadB_K:
      for (int k = 0; k < kSizeK; ++k) {

      ReadB_BufferB_M1:
        for (int m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          pipe.Push(memory[IndexB(k, m0, m1m)]); 
        }

      }
    }
  }
}

void ConvertWidthB(Stream<MemoryPackM_t, 2 * kOuterTileSizeM> &wide,
                   Stream<ComputePackM_t> &narrow) {
  // This assertion will be relaxed once Xilinx IP memory converters have been
  // inserted
  static_assert(kMemoryWidthM % kComputeTileSizeM == 0,
                "Memory width in M must be divisable by compute tile size.");

  static_assert(((kTotalReadsFromB / kMemoryWidthM) *
                 MemoryPackM_t::kWidth) == kTotalReadsFromB,
                "Sanity check failed for ConvertWidthB");

  static_assert(((kTotalReadsFromB / kMemoryWidthM) *
                 (kMemoryWidthM / kComputeTileSizeM) *
                 ComputePackM_t::kWidth) == kTotalReadsFromB,
                "Sanity check failed for ConvertWidthB");

ConvertWidthB_Outer:
  for (int i = 0; i < kTotalReadsFromB / kMemoryWidthM; ++i) {
    MemoryPackM_t memoryPack;
  ConvertWidthB_Memory:
    for (int j = 0; j < kMemoryWidthM / kComputeTileSizeM; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      if (j == 0) {
        memoryPack = wide.Pop();
      }
      ComputePackM_t computePack;
    ConvertWidthB_Compute:
      for (int w = 0; w < kComputeTileSizeM; ++w) {
        #pragma HLS UNROLL
        computePack[w] = memoryPack[j * kComputeTileSizeM + w];
      }
      narrow.Push(computePack);
    }
  }
}

void ConvertWidthC(Stream<ComputePackM_t> &narrow,
                   Stream<MemoryPackM_t> &wide) {
  static_assert(kMemoryWidthM % ComputePackM_t::kWidth == 0,
                "Memory width must be divisable by compute tile width.");

  static_assert((((kSizeN * kSizeM) / MemoryPackM_t::kWidth) *
                 (kMemoryWidthM / ComputePackM_t::kWidth) *
                 ComputePackM_t::kWidth) == kSizeN * kSizeM,
                "Sanity check failed for ConvertWidthC");

ConvertWidthC_Outer:
  for (int i = 0; i < (kSizeN * kSizeM) / MemoryPackM_t::kWidth; ++i) {
  ConvertWidthB_Memory:
    MemoryPackM_t memoryPack;
    for (int j = 0; j < kMemoryWidthM / ComputePackM_t::kWidth; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      const auto computePack = narrow.Pop();
    ConvertWidthB_Compute:
      for (int w = 0; w < ComputePackM_t::kWidth; ++w) {
        #pragma HLS UNROLL
        memoryPack[j * ComputePackM_t::kWidth + w] = computePack[w];
      }
      if (j == kMemoryWidthM / ComputePackM_t::kWidth - 1) {
        wide.Push(memoryPack);
      }
    }
  }
}

void WriteC(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[]) {

  static_assert(
      (kOuterTilesN * kOuterTilesM * kOuterTileSizeN * kOuterTileSizeMMemory *
       MemoryPackM_t::kWidth) == kSizeN * kSizeM,
      "Sanity check failed for WriteC");

WriteC_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  WriteC_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    WriteC_N1:
      for (int n1 = 0; n1 < kOuterTileSizeN; ++n1) {
      WriteC_M1:
        for (int m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          memory[IndexC(n0, n1, m0, m1m)] = pipe.Pop();
        }
      }
#ifndef MM_SYNTHESIS
      std::cout << "Finished tile (" << n0 << ", " << m0 << ") of ("
                << kOuterTilesN - 1 << ", " << kOuterTilesM - 1 << ")\n"
                << std::flush;
#endif
    }
  }
}

#ifndef MM_CONVERT_B
void FeedB(Stream<ComputePackM_t, 2 * kOuterTileSizeM> &fromMemory,
           Stream<ComputePackM_t, kPipeDepth> &toKernel) {
#else
void FeedB(Stream<ComputePackM_t> &fromMemory,
           Stream<ComputePackM_t, kPipeDepth> &toKernel) {
#endif

  static_assert(static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                        kSizeK * kInnerTilesM * ComputePackM_t::kWidth ==
                    kTotalReadsFromB,
                "Sanity check failed");

FeedB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FeedB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FeedB_K:
      for (int k = 0; k < kSizeK; ++k) {

        ComputePackM_t buffer[kInnerTilesM];

      FeedB_Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedB_Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            ComputePackM_t val;
            if (n1 == 0) {
              val = fromMemory.Pop();
              buffer[m1] = val;
            } else {
              val = buffer[m1];
            }
            toKernel.Push(val);
          }
        }

      }
    }
  }
}
