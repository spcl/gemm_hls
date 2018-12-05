/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "MatrixMultiplication.h"
#include "Memory.h"

using hlslib::Stream;

void ProcessingElement(Stream<ComputePackN_t, kPipeDepth> &aIn,
                       Stream<ComputePackN_t, kPipeDepth> &aOut,
                       Stream<ComputePackM_t, kPipeDepth> &bIn,
                       Stream<ComputePackM_t, kPipeDepth> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn,
                       const int locationN) {
  static_assert(
      (static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kSizeK *
       kInnerTilesN * kInnerTilesM * kComputeTileSizeN * kComputeTileSizeM) ==
          ((static_cast<unsigned long>(kSizeN) * kSizeK * kSizeM) /
           kComputeTilesN),
      "Sanity check for ProcessingElement failed");

  // A is double-buffered, such that new values can be read while the 
  // previous outer product is being computed. This is required to achieve
  // a perfect pipeline across the K-dimension, which is necessary for
  // many processing elements (kInnerTileSizeN).
  ComputePackN_t aBuffer[2 * kInnerTilesN];

  // This is where we spend all our T^2 fast memory
  ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
  #pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2

  // Populate the buffer for the first outer product 
InitializeABuffer_Inner:
  for (int n2 = 0; n2 < kInnerTilesN; ++n2) {
    if (locationN < kComputeTilesN - 1) {
      // All but the last processing element 
    InitializeABuffer_Outer:
      for (int n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN
        const auto read = aIn.Pop();
        if (n1 == 0) {
          aBuffer[n2] = read;
        } else {
          aOut.Push(read);
        }
      }
    } else {
      // Last processing element gets a special case, because Vivado HLS
      // refuses to flatten and pipeline loops with trip count 1
      #pragma HLS PIPELINE II=1
      aBuffer[n2] = aIn.Pop();
    }
  }

OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {

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

            static_assert(kInnerTilesM >= kInnerTilesN,
                          "Double buffering does not work if there are more "
                          "N-tiles than M-tiles");

            // Double-buffering scheme. This hijacks the m1-index to perform
            // the buffering and forwarding of values for the following outer
            // product, required to flatten the K-loop.
            if ((n0 < kOuterTilesN - 1 || m0 < kOuterTilesM - 1 ||
                 k < kSizeK - 1) &&
                m1 >= locationN            // Start at own index.
                && m1 < kComputeTilesN) {  // Number of PEs in front.
              const auto read = aIn.Pop();
              if (m1 == locationN) {
                // Double buffering
                aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
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
            const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
            #pragma HLS DEPENDENCE variable=aBuffer false
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
  Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  Stream<Data_t> aConvert("aConvert");
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];

  Stream<MemoryPack_t, 2 * kOuterTileSizeM> bMemory("bMemory");
  Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];

  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];

#ifndef HLSLIB_SYNTHESIS
  // Name the arrays of channels for debugging purposes
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
    Stream<ComputePackM_t> bFeed("bFeed");
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
    Stream<MemoryPack_t> cMemory("cMemory");
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c);
#else
    HLSLIB_DATAFLOW_FUNCTION(WriteC, cPipes[0], c);
#endif

  HLSLIB_DATAFLOW_FINALIZE();
}
