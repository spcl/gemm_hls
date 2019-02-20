/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "MatrixMultiplication.h"
#include "Memory.h"
#include <cassert>

using hlslib::Stream;

void ProcessingElement(Stream<ComputePackN_t, kPipeDepth> &aIn,
                       Stream<ComputePackN_t, kPipeDepth> &aOut,
                       Stream<ComputePackM_t, kPipeDepth> &bIn,
                       Stream<ComputePackM_t, kPipeDepth> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m) {

  assert((static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) * size_k *
          kInnerTilesN * kInnerTilesM * kComputeTileSizeN *
          kComputeTileSizeM) ==
         ((static_cast<unsigned long>(size_n) * size_k * size_m) /
          kComputeTilesN));

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
  for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2) {
    if (locationN < kComputeTilesN - 1) {
      // All but the last processing element 
    InitializeABuffer_Outer:
      for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
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
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {

      // We do not tile K further, but loop over the entire outer tile here
    Collapse_K:
      for (unsigned k = 0; k < size_k; ++k) {
        // Begin outer tile ---------------------------------------------------

      Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {

        Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {

            // Begin compute tile ---------------------------------------------
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN

            static_assert(kInnerTilesM >= kInnerTilesN,
                          "Double buffering does not work if there are more "
                          "N-tiles than M-tiles");

            // Double-buffering scheme. This hijacks the m1-index to perform
            // the buffering and forwarding of values for the following outer
            // product, required to flatten the K-loop.
            if ((n0 < OuterTilesN(size_n) - 1 || m0 < OuterTilesM(size_m) - 1 ||
                 k < size_k - 1) &&
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
            for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
              #pragma HLS UNROLL

              ComputePackM_t cStore;
              const auto cPrev = (k > 0)
                                     ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                     : ComputePackM_t(static_cast<Data_t>(0));

            Unroll_M:
              for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                #pragma HLS UNROLL

                const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                MM_MULT_RESOURCE_PRAGMA(mapped);
                const auto prev = cPrev[m2];

                const auto reduced = OperatorReduce::Apply(prev, mapped);
                MM_ADD_RESOURCE_PRAGMA(reduced);
                cStore[m2] = reduced; 
                #pragma HLS DEPENDENCE variable=cBuffer false
              }

              cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
            }

            // End compute tile -----------------------------------------------
          }
        }

        // End outer tile -----------------------------------------------------
      }

      // Write back tile of C -------------------------------------------------
      // 
      // This uses a flattened implementation of the loops, as we otherwise
      // introduce a lot of pipeline drains, which can have a small performance
      // impact for large designs.
      //
      const unsigned writeFlattenedInner =
          (kComputeTileSizeN * kInnerTilesM +
           (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
      const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;
      ap_uint<hlslib::ConstLog2(kInnerTilesN)> n1 = 0;
      ap_uint<hlslib::ConstLog2(kComputeTileSizeN)> n2 = 0;
      ap_uint<hlslib::ConstLog2(kInnerTilesM)> m1 = 0;
      unsigned inner = 0;
    WriteC_Flattened:
      for (unsigned i = 0; i < writeFlattened; ++i) {
        #pragma HLS PIPELINE II=1
        if (inner < kComputeTileSizeN * kInnerTilesM) {
          cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
          if (m1 == kInnerTilesM - 1) {
            m1 = 0;
            if (n2 == kComputeTileSizeN - 1) {
              n2 = 0;
            } else {
              ++n2;
            }
          } else {
            ++m1;
          }
        } else {
          if (locationN < kComputeTilesN - 1) {
            cOut.Push(cIn.Pop());
          }
        }
        if (inner == writeFlattenedInner - 1) {
          inner = 0;
          ++n1;
        } else {
          ++inner;
        }
      }

      // Non-flattened implementation below
    // WriteC_N1:
    //   for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
    //   WriteC_N2:
    //     for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
    //     WriteC_M1:
    //       for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_FLATTEN
    //         cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
    //       }
    //     }
    //
    //     // Forward other tiles of C
    //     // ----------------------------------------------
    //     // We send values upwards, so first tile forwards N-1 times, and the
    //     // last tile forwards 0 times.
    //     if (locationN < kComputeTilesN - 1) {
    //     ForwardC_Others:
    //       for (unsigned l = 0; l < kComputeTilesN - locationN - 1; ++l) {
    //       ForwardC_N2:
    //         for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
    //         ForwardC_M1:
    //           for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
    //             #pragma HLS PIPELINE II=1
    //             #pragma HLS LOOP_FLATTEN
    //             cOut.Push(cIn.Pop());
    //           }
    //         }
    //       }
    //     }
    //   }

      // -----------------------------------------------------------------------
    }
  }
}

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
) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
#ifdef MM_DYNAMIC_SIZES
  #pragma HLS INTERFACE s_axilite port=size_n bundle=control
  #pragma HLS INTERFACE s_axilite port=size_k bundle=control
  #pragma HLS INTERFACE s_axilite port=size_m bundle=control
#endif
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  #pragma HLS DATAFLOW

#ifndef MM_DYNAMIC_SIZES
  const unsigned size_n = kSizeN;
  const unsigned size_k = kSizeK;
  const unsigned size_m = kSizeM;
#endif

  // Memory accesses and pipes for A 
#ifndef MM_TRANSPOSED_A
  Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  Stream<Data_t> aConvert("aConvert");
#else
  Stream<MemoryPackN_t, 2 * kOuterTileSizeNMemory> aMemory("aMemory");
#endif
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];

  // Memory accesses and pipes for B 
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory("bMemory");
  Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];

  // Pipes for C
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];

#ifndef HLSLIB_SYNTHESIS
  // Name the arrays of channels for debugging purposes
#ifndef MM_TRANSPOSED_A
  for (unsigned i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
#endif
  for (unsigned n = 0; n < kComputeTilesN; ++n) {
    aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    bPipes[n].set_name(("bPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  // Only convert memory width if necessary
#ifndef MM_TRANSPOSED_A
  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit, size_n, size_k, size_m);
#ifdef MM_CONVERT_A
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aConvert);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthA, aConvert, aPipes[0]);
#else
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0], size_n, size_k,
                           size_m);
#endif
#else
  HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aMemory, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthATransposed, aMemory, aPipes[0], size_n,
                           size_k, size_m);
#endif

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);

    // Only convert memory width if necessary
#ifdef MM_CONVERT_B
  Stream<ComputePackM_t> bFeed("bFeed");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0], size_n, size_k, size_m);
#endif

  for (unsigned pe = 0; pe < kComputeTilesN; ++pe) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                             aPipes[pe],
                             aPipes[pe + 1],
                             bPipes[pe],
                             bPipes[pe + 1],
                             cPipes[pe],
                             cPipes[pe + 1],
                             pe, size_n, size_k, size_m);
  }

  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory("cMemory");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}
