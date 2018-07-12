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

void DistributeA(Stream<ComputePackN_t> &fromMemory,
                 Stream<ComputePackN_t> toFeeders[kComputeTilesN]) {

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 kSizeK * kInnerTilesN * kComputeTilesN *
                 ComputePackN_t::kWidth) == kTotalReadsFromA,
                "Sanity check failed for DistributeA");

DistributeA_N0:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  DistributeA_M0:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    DistributeA_K:
    for (int k = 0; k < kSizeK; ++k) {
      DistributeA_N1:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        DistributeA_N2:
          for (int n2 = 0; n2 < kComputeTilesN; ++n2) { 
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            toFeeders[n2 * (kComputeTilesM + 2)].Push(fromMemory.Pop());
          }
        }
      }
    }
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
    const auto memoryPack = wide.Pop();
  ConvertWidthB_Memory:
    for (int j = 0; j < kMemoryWidth / kComputeTileSizeM; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
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

void DistributeB(Stream<ComputePackM_t> &pipe,
                 Stream<ComputePackM_t> toFeeders[kComputeTilesM]) {

  static_assert((static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM *
                 kSizeK * kInnerTilesM * kComputeTilesM *
                 ComputePackM_t::kWidth) == kTotalReadsFromB,
                "Sanity check failed for DistributeB");

DistributeB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  DistributeB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    DistributeB_K:
      for (int k = 0; k < kSizeK; ++k) {
      DistributeB_M1:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
        DistributeB_M2:
          for (int m2 = 0; m2 < kComputeTilesM; ++m2) { 
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            toFeeders[m2].Push(pipe.Pop()); 
          }
        }
      }
    }
  }
}

void FanInC(Stream<OutputPack_t> fromDrainers[kComputeTilesM],
            Stream<OutputPack_t> &toMemory) {

  static_assert((kOuterTilesN * kOuterTilesM * kInnerTilesN * kInnerTilesM *
                 kComputeTilesN * kComputeTilesM * OutputPack_t::kWidth) ==
                kSizeN * kSizeM, "Sanity check failed for FanInC");

FanInC_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FanInC_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FanInC_N1:
      for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
      FanInC_M1:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
        FanInC_N2:
          for (int n2 = 0; n2 < kComputeTilesN; ++n2) {
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
}

void ConvertWidthC(Stream<OutputPack_t> &narrow, Stream<MemoryPack_t> &wide) {

  // TODO: fix when output is wider than memory
  static_assert(kMemoryWidth % OutputPack_t::kWidth == 0,
                "Memory width must be divisable by compute tile width.");

  static_assert((((kSizeN * kSizeM) / OutputPack_t::kWidth) *
                 (kMemoryWidth / OutputPack_t::kWidth) *
                 OutputPack_t::kWidth) == kSizeN * kSizeM,
                "Sanity check failed for ConvertWidthC");

ConvertWidthC_Outer:
  for (int i = 0; i < (kSizeN * kSizeM) / OutputPack_t::kWidth; ++i) {
  ConvertWidthB_Memory:
    const auto memoryPack = wide.Pop();
    for (int j = 0; j < kMemoryWidth / OutputPack_t::kWidth; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      OutputPack_t computePack;
    ConvertWidthB_Compute:
      for (int w = 0; w < OutputPack_t::kWidth; ++w) {
        #pragma HLS UNROLL
        computePack[w] = memoryPack[j * OutputPack_t::kWidth + w];
      }
      narrow.Push(computePack);
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
