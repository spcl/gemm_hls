/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
#include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

using hlslib::Stream;
using hlslib::DataPack;

/// Feeds a single compute row
void FeedA(Stream<ComputePackN_t> &fromMemory,
           Stream<ComputePackN_t> &toKernel) {
FeedA_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FeedA_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FeedA_OuterTile_K:
      for (int k0 = 0; k0 < kOuterTilesK; ++k0) {
        ComputePackN_t buffer[kInnerTilesN * kOuterTileSize];
      FeedA_Pipeline_K:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        FeedA_Pipeline_N:
          for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
          FeedA_Pipeline_M:
            for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              ComputePackN_t pack;
              const auto index = k1 * kOuterTileSize + n1;
              if (k0 == 0 && m1 == 0) {
                pack = fromMemory.Pop();
                buffer[index] = pack;
              } else {
                pack = buffer[index];
              }
              toKernel.Push(pack);
            }
          }
        }
      }
    }
  }
}

/// Feeds a single compute column
void FeedB(Stream<ComputePackM_t> &fromMemory,
           Stream<ComputePackM_t> &toKernel) {
FeedB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  FeedB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    FeedB_OuterTile_K:
      for (int k0 = 0; k0 < kOuterTilesK; ++k0) {
        ComputePackM_t buffer[kInnerTilesM * kOuterTileSize];
      FeedB_Pipeline_K:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        FeedB_Pipeline_N:
          for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
          FeedB_Pipeline_M:
            for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              ComputePackM_t pack;
              const auto index = k1 * kOuterTileSize + m1;
              if (k0 == 0 && n1 == 0) {
                pack = fromMemory.Pop();
                buffer[index] = pack;
              } else {
                pack = buffer[index];
              }
              toKernel.Push(pack);
            }
          }
        }
      }
    }
  }
}

int IndexCBuffer(int n1, int n2, int m1, int m2) {
  #pragma HLS INLINE
  return (n1 * kInnerTileSizeN + n2) * kOuterTileSize +
         (m1 * kInnerTileSizeM + m2);
}

void ProcessingElement(Stream<ComputePackN_t> &aIn, Stream<ComputePackN_t> &aOut,
                   Stream<ComputePackM_t> &bIn, Stream<ComputePackM_t> &bOut,
                   Stream<OutputPack_t> &cIn, Stream<OutputPack_t> &cOut,
                   const int locationN, const int locationM) {
OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {

      Data_t cBuffer[kInnerTilesN * kComputeTileSizeN * kInnerTilesM *
                     kComputeTileSizeM];
      #pragma HLS ARRAY_PARTITION variable=cBuffer cyclic factor=kComputeTileSizeN*kComputeTileSizeM

    OuterTile_K:
      for (int k0 = 0; k0 < kOuterTilesK; ++k0) {
        // Begin outer tile ---------------------------------------------------

        // We do not tile M further, but loop over the entire outer tile here
      Pipeline_K:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        Pipeline_N:
          for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
          Pipeline_M:
            for (int m1 = 0; m1 < kInnerTilesM; ++m1) {

              // Begin compute tile --------------------------------------------
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN

              const auto aVal = aIn.Pop();
              if (locationN < kComputeTilesN - 1) {
                aOut.Push(aVal);
              }
              const auto bVal = bIn.Pop();
              if (locationM < kComputeTilesM - 1) {
                bOut.Push(bVal);
              }
            
            Unroll_N:
              for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
                #pragma HLS UNROLL

              Unroll_M:
                for (int m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                  #pragma HLS UNROLL

                  const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);

                  const auto prev = (k0 == 0 && k1 == 0)
                                        ? 0
                                        : cBuffer[IndexCBuffer(n1, n2, m1, m2)];
                  cBuffer[IndexCBuffer(n1, n2, m1, m2)] =
                      OperatorReduce::Apply(prev, mapped);
                  #pragma HLS DEPENDENCE variable=cBuffer false
                }
              }

              // End compute tile ----------------------------------------------
            }
          }
        }
        // End outer tile -----------------------------------------------------
      }

      // Forward other tiles of C ----------------------------------------------
      // We send values upwards, so first tile forwards N-1 times, and the
      // last time forwards 0 times.
    ForwardC_Others:
      for (int l = 0; l < kComputeTilesN - locationN - 1; ++l) {
      ForwardC_N1:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        ForwardC_M1:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            cOut.Push(cIn.Pop());
          }
        }
      }
      // -----------------------------------------------------------------------

      // Write back own tile of C ----------------------------------------------
    WriteC_N1:
      for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
      WriteC_M1:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {

          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN

          OutputPack_t pack;

        WriteC_N2:
          for (int n2 = 0; n2 < kComputeTileSizeN; ++n2) {
            #pragma HLS UNROLL
          WriteC_M2:
            for (int m2 = 0; m2 < kComputeTileSizeM; ++m2) {
              #pragma HLS UNROLL
              pack[n2 * kComputeTileSizeM + m2] =
                  cBuffer[IndexCBuffer(n1, n2, m1, m2)];
            }
          }
          
          cOut.Push(pack);
        }
      }
      // -----------------------------------------------------------------------

    }
  }

}

void MatrixMatrix(MemoryPack_t const a[], MemoryPack_t const b[],
                  MemoryPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmek0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmek1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  Stream<Data_t> aSplit[kMemoryWidth];
  Stream<Data_t> aConvert("aConvert");
  Stream<ComputePackN_t> aDistribute("aDistribute");
  Stream<ComputePackN_t> aPipes[kComputeTilesN * (kComputeTilesM + 1)];

  Stream<MemoryPack_t> bMemory("bMemory");
  Stream<ComputePackM_t> bDistribute("bDistribute");
  Stream<ComputePackM_t> bPipes[(kComputeTilesN + 1) * kComputeTilesM];
  Stream<OutputPack_t> cPipes[(kComputeTilesN + 1) * kComputeTilesM];
  Stream<OutputPack_t> cConvert("cConvert");
  Stream<MemoryPack_t> cMemory("cMemory");

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
    HLSLIB_DATAFLOW_FUNCTION(FeedA, aPipes[n * kComputeTilesM],
                             aPipes[(n + 1) * kComputeTilesM]);
  }

  for (int m = 0; m < kComputeTilesM; ++m) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(FeedB, bPipes[m], bPipes[m + 1]);
  }

  for (int n = 0; n < kComputeTilesN; ++n) {
    #pragma HLS UNROLL
    for (int m = 0; m < kComputeTilesM; ++m) {
      #pragma HLS UNROLL
      HLSLIB_DATAFLOW_FUNCTION(
          ProcessingElement, aPipes[n * (kComputeTilesM + 1) + m + 1],
          aPipes[n * (kComputeTilesM + 1) + m + 2],
          bPipes[(n + 1) * kComputeTilesM + m],
          bPipes[(n + 2) * kComputeTilesM + m], cPipes[n * kComputeTilesM + m],
          cPipes[(n + 1) * kComputeTilesM + m], n, m);
    }
  }

  HLSLIB_DATAFLOW_FUNCTION(FanInC, &cPipes[kComputeTilesN * kComputeTilesM],
                           cConvert);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cConvert, cMemory);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c);

  HLSLIB_DATAFLOW_FINALIZE();
}
