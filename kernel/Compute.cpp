/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "MatrixMultiplication.h"
#include "Memory.h"
#include <cassert>

using hlslib::Stream;

int IndexCBuffer(int n1, int n2, int m1, int m2) {
  #pragma HLS INLINE
  auto index = (n1 * kComputeTileSizeN + n2) * kComputeTileSizeM +
               (m1 * kComputeTileSizeM + m2);
  assert(index <
         kInnerTilesN * kComputeTileSizeN * kInnerTilesM * kComputeTileSizeM);
  return index;
}

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const int locationN) {

  static_assert(
      (static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kSizeK *
       kInnerTilesN * kInnerTilesM * kComputeTileSizeN * kComputeTileSizeM) ==
          ((static_cast<unsigned long>(kSizeN) * kSizeK * kSizeM) /
           kComputeTilesN),
      "Sanity check for ProcessingElement failed");

  const int kInitializePhase = kInnerTilesN * (kComputeTilesN - locationN);
  constexpr int kComputePhase = kSizeK * kInnerTilesN * kInnerTilesM;
  constexpr int kDrainPhaseOwn = kComputeTileSizeN * kInnerTilesM;
  const int kDrainPhaseOthers =
      (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM;
  const int kDrainPhaseInner = kDrainPhaseOwn + kDrainPhaseOthers;
  const int kDrainPhase = kInnerTilesN * kDrainPhaseInner;

  int i_innerTilesN = 0;
  const int i_innerTilesN_max = kInnerTilesN;

  int i_innerTilesM = 0;
  const int i_innerTilesM_max = kInnerTilesM;

  int i_computeTilesN = 0;
  const int i_computeTilesN_max = kComputeTilesN - locationN;

  int i_computeTileSizeN = 0;
  const int i_computeTileSizeN_max = kComputeTileSizeN;

  int i_sizeK = 0;
  const int i_sizeK_max = kSizeK;

  int i_drainPhase = 0;
  const int i_drainPhase_max = kDrainPhaseInner; 

OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {

    FlattenedLoops:
      for (int i = 0; i < kInitializePhase + kComputePhase + kDrainPhase; ++i) {

        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN

        // A is double-buffered, such that new values can be read while the
        // previous outer product is being computed. This is required to achieve
        // a perfect pipeline across the K-dimension, which is necessary for
        // many processing elements (kInnerTileSizeN).
        ComputePackN_t aBuffer[2 * kInnerTilesN];

        ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
        #pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2
    
        //---------------------------------------------------------------------
        // Begin initialize phase
        if (i < kInitializePhase) {

          const auto read = aIn.Pop();

          if (i_computeTilesN == 0) {
            aBuffer[i_innerTilesN] = read;
          } else {
            // Put extra condition here, otherwise Vivado HLS cannot infer that
            // this never happens
            if (locationN < kComputeTilesN - 1) {
              aOut.Push(read);
            }
          }

          // Update indices
          if (i_computeTilesN == i_computeTilesN_max - 1) {
            i_computeTilesN = 0;
            if (i_innerTilesN == i_innerTilesN_max - 1) {
              i_innerTilesN = 0;
            } else {
              ++i_innerTilesN;
            }
          } else {
            ++i_computeTilesN;
          }

          // End initialize phase
          //-------------------------------------------------------------------
        } else if (i < kInitializePhase + kComputePhase) {
          //-------------------------------------------------------------------
          // Begin compute phase 

          static_assert(kInnerTilesM >= kInnerTilesN,
                        "Double buffering does not work if there are more "
                        "N-tiles than M-tiles");

          // Double-buffering scheme. This hijacks the i_innerTilesM-index to
          // perform the buffering and forwarding of values for the following
          // outer product, required to flatten the K-loop.
          if (i_sizeK < kSizeK - 1  // Don't forward on the last iteration.
              && i_innerTilesM >= locationN  // Start at own index.
              && i_innerTilesM < kComputeTilesN) {  // Number of PEs in front. 

            const auto read = aIn.Pop();
            if (i_innerTilesM == locationN) {
              // Double buffering
              aBuffer[i_innerTilesN + (i_sizeK % 2 == 0 ? kInnerTilesN : 0)] =
                  read;
              #pragma HLS DEPENDENCE variable=aBuffer false
            } else {
              // Without this check, Vivado HLS thinks aOut can be written
              // from the last processing element and fails dataflow
              // checking.
              if (locationN < kComputeTilesN - 1) {
                aOut.Push(read);
              }
            }
          }

          // Double buffering, read from the opposite end of where the buffer
          // is being written
          const auto aVal =
              aBuffer[i_innerTilesN + (i_sizeK % 2 == 0 ? 0 : kInnerTilesN)];
          #pragma HLS DEPENDENCE variable=aBuffer false

          const auto bVal = bIn.Pop();
          if (locationN < kComputeTilesN - 1) {
            bOut.Push(bVal);
          }

        Unroll_N:
          for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
            #pragma HLS UNROLL

            ComputePackM_t cStore;
            const auto cPrev =
                (i_sizeK > 0)
                    ? cBuffer[i_innerTilesN * kInnerTilesM + i_innerTilesM][n2]
                    : ComputePackM_t(static_cast<Data_t>(0));

          Unroll_M:
            for (int m2 = 0; m2 < kComputeTileSizeM; ++m2) {
              #pragma HLS UNROLL

              const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
              const auto prev = cPrev[m2];

              cStore[m2] = OperatorReduce::Apply(prev, mapped);
              #pragma HLS DEPENDENCE variable=cBuffer false
            }

            cBuffer[i_innerTilesN * kInnerTilesM + i_innerTilesM][n2] = cStore;
          }

          if (i_innerTilesM == i_innerTilesM_max - 1) {
            i_innerTilesM = 0;
            if (i_innerTilesN == i_innerTilesN_max - 1) {
              i_innerTilesN = 0;
              if (i_sizeK == i_sizeK_max - 1) {
                i_sizeK = 0;
              } else {
                ++i_sizeK;
              }
            } else {
              ++i_innerTilesN;
            }
          } else {
            ++i_innerTilesM;
          }

          // End compute phase 
          //-------------------------------------------------------------------
        } else {
          //-------------------------------------------------------------------
          
          if (i_drainPhase < kDrainPhaseOwn) {
            //-----------------------------------------------------------------
            // Begin draining own tile 

            cOut.Push(cBuffer[i_innerTilesN * kInnerTilesM + i_innerTilesM]
                             [i_computeTileSizeN]);

            if (i_innerTilesM == i_innerTilesM_max - 1) {
              i_innerTilesM = 0;
              if (i_computeTileSizeN == i_computeTileSizeN_max - 1) {
                i_computeTileSizeN = 0;
              } else {
                ++i_computeTileSizeN;
              }
            } else {
              ++i_innerTilesM;
            }

            // End draining own tile 
            //-----------------------------------------------------------------
          } else {
            //-----------------------------------------------------------------
            // Begin forwarding other tiles 

            // Need an extra condition so Vivado HLS understands that this is
            // never written in the last PE
            if (locationN < kComputeTilesN - 1) {
              cOut.Push(cIn.Pop());
            }

            // End forwarding other tiles 
            //-----------------------------------------------------------------
          }

          if (i_drainPhase == i_drainPhase_max - 1) {
            i_drainPhase = 0;
            if (i_innerTilesN == i_innerTilesN_max - 1) {
              i_innerTilesN = 0;
            } else {
              ++i_innerTilesN;
            }
          } else {
            ++i_drainPhase;
          }

          // End drain phase 
          //-------------------------------------------------------------------
        }

      } // End flattened loop

    } // End outer tile M
  } // End outer tile N

}

void MatrixMultiplicationKernel(MemoryPack_t const a[], MemoryPack_t const b[],
                                MemoryPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW
  // TODO: does this need to be kOuterTileSizeN?
  Stream<Data_t, kOuterTileSizeN> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=kOuterTileSizeN
  Stream<Data_t> aConvert("aConvert");
  Stream<ComputePackN_t> aPipes[kComputeTilesN + 1];

  Stream<MemoryPack_t> bMemory("bMemory");
  Stream<ComputePackM_t> bDistribute("bDistribute");
  Stream<ComputePackM_t> bPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t> bFeed("bFeed");

  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];
  Stream<MemoryPack_t> cMemory("cMemory");

#ifndef HLSLIB_SYNTHESIS
  for (int i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
  for (int n = 0; n < kComputeTilesN; ++n) {
    aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
  }
  for (int n = 0; n < kComputeTilesN + 1; ++n) {
    bPipes[n].set_name(("bPipes[" + std::to_string(n) + "]").c_str());
  }
  for (int n = 0; n < kComputeTilesN + 1; ++n) {
    cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit);

  // Only convert memory width if necessary
#ifdef MM_CONVERT_A
    HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aConvert);
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthA, aConvert, aPipes[0]);
#else
    HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0]);
#endif

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory);

  // Only convert memory width if necessary
#ifdef MM_CONVERT_B
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed);
    HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0]);
#else
    HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0]);
#endif

  for (int n = 0; n < kComputeTilesN; ++n) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                             aPipes[n],
                             aPipes[n + 1],
                             bPipes[n],
                             bPipes[n + 1],
                             cPipes[n],
                             cPipes[n + 1],
                             n);
  }

  // Only convert memory width if necessary
#ifdef MM_CONVERT_B
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c);
#else
    HLSLIB_DATAFLOW_FUNCTION(WriteC, cPipes[0], c);
#endif

  HLSLIB_DATAFLOW_FINALIZE();
}
