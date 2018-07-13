/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
#include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include <cassert>

using hlslib::Stream;
using hlslib::DataPack;

/// Feeds a single compute row
void FeedA(Stream<ComputePackN_t> &fromMemory,
           Stream<ComputePackN_t> &toKernel) {

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

      FeedA_Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedA_Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN

            ComputePackN_t pack;
            if (m1 == 0) {
              pack = fromMemory.Pop();
              buffer[n1] = pack;
            } else {
              pack = buffer[n1];
            }
            toKernel.Push(pack);
          }
        }

      }
    }
  }
}

/// Feeds a single compute column
void FeedB(Stream<ComputePackM_t> &fromMemory,
           Stream<ComputePackM_t> &toKernel) {

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

      FeedB_Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedB_Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            ComputePackM_t pack;
            if (n1 == 0) {
              pack = fromMemory.Pop();
              buffer[m1] = pack;
            } else {
              pack = buffer[m1];
            }
            toKernel.Push(pack);
          }
        }

      }
    }
  }
}

int IndexCBuffer(int n1, int n2, int m1, int m2) {
  #pragma HLS INLINE
  auto index =
      (n1 * kComputeTileSizeN + n2) * kComputeTileSizeM * kComputeTilesM +
      (m1 * kComputeTileSizeM + m2);
  assert(index <
         kInnerTilesN * kComputeTileSizeN * kInnerTilesM * kComputeTileSizeM);
  return index;
}

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cIn,
                       Stream<ComputePackM_t> &cOut,
                       const int locationN,
                       const int locationM) {

  static_assert(
      (static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kSizeK *
       kInnerTilesN * kInnerTilesM * kComputeTileSizeN * kComputeTileSizeM) ==
          ((static_cast<unsigned long>(kSizeN) * kSizeK * kSizeM) /
           (kComputeTilesN * kComputeTilesM)),
      "Sanity check for ProcessingElement failed");

OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {

      ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];

      // We do not tile K further, but loop over the entire outer tile here
    Collapse_K:
      for (int k = 0; k < kSizeK; ++k) {
        // Begin outer tile ---------------------------------------------------

      Pipeline_N:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        Pipeline_M:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {

            // Begin compute tile --------------------------------------------
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN

            const auto aVal = aIn.Pop();
            if (locationM < kComputeTilesM - 1) {
              aOut.Push(aVal);
            }
            const auto bVal = bIn.Pop();
            if (locationN < kComputeTilesN - 1) {
              bOut.Push(bVal);
            }

          Unroll_N:
            for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
              #pragma HLS UNROLL

              ComputePackM_t cStore;
              const auto cPrev = (k > 0)
                                     ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                     : ComputePackM_t(static_cast<Data_t>(0));

            Unroll_M:
              for (int m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                #pragma HLS UNROLL

                const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                const auto prev = cPrev[m2];

                cStore[m2] = OperatorReduce::Apply(prev, mapped);
                #pragma HLS DEPENDENCE variable=cBuffer false
              }

              cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
            }

            // End compute tile ----------------------------------------------
          }
        }

        // End outer tile -----------------------------------------------------
      }

      // Write back tile of C --------------------------------------------------
    WriteC_N1:
      for (int n1 = 0; n1 < kInnerTilesN; ++n1) {

      WriteC_N2:
        for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
        WriteC_M1:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
          }
        }

        // Forward other tiles of C ----------------------------------------------
        // We send values upwards, so first tile forwards N-1 times, and the
        // last tile forwards 0 times.
        if (locationN < kComputeTilesN - 1) {
        ForwardC_Others:
          for (int l = 0; l < kComputeTilesN - locationN - 1; ++l) {
          ForwardC_N2:
            for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
            ForwardC_M1:
              for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                cOut.Push(cIn.Pop());
              }
            }
          }
        }

      }
      // -----------------------------------------------------------------------

    }
  }

}

void MatrixMatrix(MemoryPack_t const a[], MemoryPack_t const b[],
                  MemoryPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  // TODO: does this need to be kOuterTileSize?
  Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=kOuterTileSize
  Stream<Data_t> aConvert("aConvert");
  Stream<ComputePackN_t> aDistribute("aDistribute");
  Stream<ComputePackN_t> aPipes[kComputeTilesN * (kComputeTilesM + 2)];

  Stream<MemoryPack_t> bMemory("bMemory");
  Stream<ComputePackM_t> bDistribute("bDistribute");
  Stream<ComputePackM_t> bPipes[(kComputeTilesN + 2) * kComputeTilesM];
  Stream<ComputePackM_t> cPipes[(kComputeTilesN + 1) * kComputeTilesM];
  Stream<ComputePackM_t> cConvert("cConvert");
  Stream<MemoryPack_t> cMemory("cMemory");

#ifndef HLSLIB_SYNTHESIS
  for (int i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
  for (int n = 0; n < kComputeTilesN; ++n) {
    for (int m = 0; m < kComputeTilesM + 2; ++m) {
      aPipes[n * (kComputeTilesM + 2) + m].set_name(
          ("aPipes[" + std::to_string(n) + "][" + std::to_string(m) + "]")
              .c_str());
    }
  }
  for (int n = 0; n < kComputeTilesN + 2; ++n) {
    for (int m = 0; m < kComputeTilesM; ++m) {
      bPipes[n * kComputeTilesM + m].set_name(
          ("bPipes[" + std::to_string(n) + "][" + std::to_string(m) + "]")
              .c_str());
    }
  }
  for (int n = 0; n < kComputeTilesN + 1; ++n) {
    for (int m = 0; m < kComputeTilesM; ++m) {
      cPipes[n * kComputeTilesM + m].set_name(
          ("cPipes[" + std::to_string(n) + "][" + std::to_string(m) + "]")
              .c_str());
    }
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit);
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aConvert);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthA, aConvert, aDistribute);
  HLSLIB_DATAFLOW_FUNCTION(DistributeA, aDistribute, aPipes);

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bDistribute);
  HLSLIB_DATAFLOW_FUNCTION(DistributeB, bDistribute, bPipes);

  for (int n = 0; n < kComputeTilesN; ++n) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(FeedA, aPipes[n * (kComputeTilesM + 2)],
                             aPipes[n * (kComputeTilesM + 2) + 1]);
  }

  for (int m = 0; m < kComputeTilesM; ++m) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(FeedB, bPipes[m], bPipes[kComputeTilesM + m]);
  }

  for (int n = 0; n < kComputeTilesN; ++n) {
    #pragma HLS UNROLL
    for (int m = 0; m < kComputeTilesM; ++m) {
      #pragma HLS UNROLL
      HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                               aPipes[n * (kComputeTilesM + 2) + m + 1],
                               aPipes[n * (kComputeTilesM + 2) + m + 2],
                               bPipes[(n + 1) * kComputeTilesM + m],
                               bPipes[(n + 2) * kComputeTilesM + m],
                               cPipes[(n + 1) * kComputeTilesM + m],
                               cPipes[n * kComputeTilesM + m], n, m);
    }
  }

  HLSLIB_DATAFLOW_FUNCTION(FanInC, cPipes, cConvert);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cConvert, cMemory);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c);

  HLSLIB_DATAFLOW_FINALIZE();
}
