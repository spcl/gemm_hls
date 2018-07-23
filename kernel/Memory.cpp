/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Memory.h"
#include <cassert>

using hlslib::Stream;

int IndexA(int n0, int n1, int n2, int k0, int k1) {
  #pragma HLS INLINE
  const auto index =
      (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeKMemory +
      (k0 * (kTransposeWidth / kMemoryWidth) + k1);
  assert(index < kSizeN * kSizeKMemory);
  return index;
}

int IndexB(int k, int m0, int m1m) {
  #pragma HLS INLINE
  const auto index = k * kSizeMMemory + (m0 * kOuterTileSizeMemory + m1m);
  assert(index < kSizeK * kSizeMMemory);
  return index;
}

int IndexC(int n0, int n1, int m0, int m1m) {
  #pragma HLS INLINE
  const auto index = (n0 * kOuterTileSize + n1) * kSizeMMemory +
                     (m0 * kOuterTileSizeMemory + m1m);
  assert(index < kSizeN * kSizeMMemory);
  return index;
}

void _ReadAInner(MemoryPack_t const a[],
                 Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth], int n0,
                 int n1, int n2, int k0, int k1) {
  #pragma HLS INLINE
  auto pack = a[IndexA(n0, n1, n2, k0, k1)];
  for (int w = 0; w < kMemoryWidth; ++w) {
    #pragma HLS UNROLL
    aSplit[k1 * kMemoryWidth + w].Push(pack[w]); 
  }
}

template <int innerReads>
void _ReadAInnerLoop(MemoryPack_t const a[],
                     Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth],
                     int n0, int n1, int n2, int k0) {
  #pragma HLS INLINE
  for (int k1 = 0; k1 < (kTransposeWidth / kMemoryWidth); ++k1) { 
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    _ReadAInner(a, aSplit, n0, n1, n2, k0, k1);
  }
}

// Need a special case for kMemoryWidth == kTransposeWidth, as Vivado HLS
// otherwise doesn't pipeline the loops (because the inner trip count is 1).
template <>
void _ReadAInnerLoop<1>(MemoryPack_t const a[],
                        Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth],
                        int n0, int n1, int n2, int k0) {
  #pragma HLS INLINE
  #pragma HLS PIPELINE II=1
  #pragma HLS LOOP_FLATTEN
  _ReadAInner(a, aSplit, n0, n1, n2, k0, 0);
}

void ReadA(MemoryPack_t const a[],
           Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth]) {

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 (kSizeK / kTransposeWidth) * kInnerTilesN * kInnerTileSizeN *
                 (kTransposeWidth / kMemoryWidth) * MemoryPack_t::kWidth) ==
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
            _ReadAInnerLoop<kTransposeWidth / kMemoryWidth>(a, aSplit, n0, n1,
                                                            n2, k0);
          }
        }
      }
    }
  }
}

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
void TransposeA(Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth],
                Stream<Data_t> &toKernel) {

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 kSizeK * kOuterTileSize) == kTotalReadsFromA,
                "Sanity check failed for TransposeA");

TransposeA_N0:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  TransposeA_M0:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    TransposeA_K:
      for (int k = 0; k < kSizeK; ++k) {
      TransposeA_N1:
        for (int n1 = 0; n1 < kOuterTileSize; ++n1) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          // Pop from each stream kOuterTileSize times in a row
          toKernel.Push(aSplit[k % kTransposeWidth].Pop());
        }
      }
    }
  }
}

void ConvertWidthA(Stream<Data_t> &narrow, Stream<ComputePackN_t> &wide) {
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

void ReadB(MemoryPack_t const memory[], Stream<MemoryPack_t> &pipe) {

  static_assert(
      (static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kSizeK *
       kOuterTileSizeMemory * MemoryPack_t::kWidth) == kTotalReadsFromB,
      "Sanity check failed for ReadB");

ReadB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  ReadB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    ReadB_K:
      for (int k = 0; k < kSizeK; ++k) {

      ReadB_BufferB_M1:
        for (int m1m = 0; m1m < kOuterTileSizeMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          pipe.Push(memory[IndexB(k, m0, m1m)]); 
        }

      }
    }
  }
}

void ConvertWidthB(Stream<MemoryPack_t> &wide, Stream<ComputePackM_t> &narrow) {

  // This assertion will be relaxed once Xilinx IP memory converters have been
  // inserted
  static_assert(kMemoryWidth % kComputeTileSizeM == 0,
                "Memory width must be divisable by compute tile size.");

  static_assert(((kTotalReadsFromB / kMemoryWidth) *
                 MemoryPack_t::kWidth) == kTotalReadsFromB,
                "Sanity check failed for ConvertWidthB");

  static_assert(((kTotalReadsFromB / kMemoryWidth) *
                 (kMemoryWidth / kComputeTileSizeM) * ComputePackM_t::kWidth) ==
                    kTotalReadsFromB,
                "Sanity check failed for ConvertWidthB");
                
ConvertWidthB_Outer:
  for (int i = 0; i < kTotalReadsFromB / kMemoryWidth; ++i) {
    MemoryPack_t memoryPack;
  ConvertWidthB_Memory:
    for (int j = 0; j < kMemoryWidth / kComputeTileSizeM; ++j) {
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

void FanInC(Stream<ComputePackM_t> fromDrainers[kComputeTilesM],
            Stream<ComputePackM_t> &toMemory) {

  static_assert((kOuterTilesN * kOuterTilesM * kOuterTileSize * kInnerTilesM *
                 kComputeTilesM * ComputePackM_t::kWidth) == kSizeN * kSizeM,
                "Sanity check failed for FanInC");

FanInC_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FanInC_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FanInC_N1:
      for (int n1 = 0; n1 < kOuterTileSize; ++n1) {
      FanInC_M1:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
        FanInC_M2:
          for (int m2 = 0; m2 < kComputeTilesM; ++m2) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            toMemory.Push(fromDrainers[m2].Pop());
          }
        }
      }
    }
  }
}

void ConvertWidthC(Stream<ComputePackM_t> &narrow, Stream<MemoryPack_t> &wide) {

  static_assert(kMemoryWidth % ComputePackM_t::kWidth == 0,
                "Memory width must be divisable by compute tile width.");

  static_assert((((kSizeN * kSizeM) / MemoryPack_t::kWidth) *
                 (kMemoryWidth / ComputePackM_t::kWidth) *
                 ComputePackM_t::kWidth) == kSizeN * kSizeM,
                "Sanity check failed for ConvertWidthC");

ConvertWidthC_Outer:
  for (int i = 0; i < (kSizeN * kSizeM) / MemoryPack_t::kWidth; ++i) {
  ConvertWidthB_Memory:
    MemoryPack_t memoryPack;
    for (int j = 0; j < kMemoryWidth / ComputePackM_t::kWidth; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      const auto computePack = narrow.Pop();
    ConvertWidthB_Compute:
      for (int w = 0; w < ComputePackM_t::kWidth; ++w) {
        #pragma HLS UNROLL
        memoryPack[j * ComputePackM_t::kWidth + w] = computePack[w];
      }
      if (j == kMemoryWidth / ComputePackM_t::kWidth - 1) {
        wide.Push(memoryPack);
      }
    }
  }
}

void WriteC(Stream<MemoryPack_t> &pipe, MemoryPack_t memory[]) {

  static_assert(
      (kOuterTilesN * kOuterTilesM * kOuterTileSize * kOuterTileSizeMemory *
       MemoryPack_t::kWidth) == kSizeN * kSizeM,
      "Sanity check failed for WriteC");

WriteC_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  WriteC_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    WriteC_N1:
      for (int n1 = 0; n1 < kOuterTileSize; ++n1) {
      WriteC_M1:
        for (int m1m = 0; m1m < kOuterTileSizeMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          memory[IndexC(n0, n1, m0, m1m)] = pipe.Pop();
        }
      }
    }
  }
}

/// Feeds a single compute row
void FeedA(Stream<ComputePackN_t> &previous,
           Stream<ComputePackN_t> &next,
           Stream<ComputePackN_t> &toKernel,
           const int locationN) {

  static_assert(static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                        kSizeK * kInnerTilesN * ComputePackN_t::kWidth ==
                    kTotalReadsFromA / kComputeTilesN,
                "Sanity check failed");

FeedA_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FeedA_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FeedA_K:
      for (int k = 0; k < kSizeK; ++k) {

        ComputePackN_t buffer[kInnerTilesN];

      FeedA_SaturateOuter:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
          if (locationN < kComputeTilesN - 1) {
          FeedA_SaturateInner:
            for (int n2 = 0; n2 < kComputeTilesN - locationN; ++n2) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              const auto read = previous.Pop();
              if (n2 == 0) {
                buffer[n1] = read;
              } else {
                next.Push(read);
              }
            }
          } else {
            // Special case is needed to:
            // 1) Workaround Vivado HLS not flattening and pipelining loops
            //    with trip count == 1.
            // 2) Convince Vivado HLS that next is never written for the last
            //    feeder.
            #pragma HLS PIPELINE II=1
            buffer[n1] = previous.Pop();
          }
        }

      FeedA_Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedA_Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            toKernel.Push(buffer[n1]);
          }
        }

      }
    }
  }
}

/// Feeds a single compute column
void FeedB(Stream<ComputePackM_t> &previous,
           Stream<ComputePackM_t> &next,
           Stream<ComputePackM_t> &toKernel, const int locationM) {

  static_assert(static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                        kSizeK * kInnerTilesM * ComputePackM_t::kWidth ==
                    kTotalReadsFromB / kComputeTilesM,
                "Sanity check failed");

FeedB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FeedB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FeedB_K:
      for (int k = 0; k < kSizeK; ++k) {

        ComputePackM_t buffer[kInnerTilesM];

      FeedB_SaturateOuter:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
          if (locationM < kComputeTilesM - 1) {
          FeedB_SaturateInner:
            for (int m2 = 0; m2 < kComputeTilesM - locationM; ++m2) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              const auto read = previous.Pop();
              if (m2 == 0) {
                buffer[m1] = read;
              } else {
                next.Push(read);
              }
            }
          } else {
            // Special case is needed to:
            // 1) Workaround Vivado HLS not flattening and pipelining loops
            //    with trip count == 1.
            // 2) Convince Vivado HLS that next is never written for the last
            //    feeder.
            #pragma HLS PIPELINE II=1
            buffer[m1] = previous.Pop();
          }
        }

      FeedB_Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedB_Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            toKernel.Push(buffer[m1]);
          }
        }

      }
    }
  }
}
