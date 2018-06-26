/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
// #include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

using hlslib::Stream;
using hlslib::DataPack;

int IndexA(int n0, int n1, int n2, int k0, int k1) {
  #pragma HLS INLINE
  return (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeK +
         (k0 * kOuterTileSize + k1);
}

int IndexB(int k0, int k1, int m0, int m1, int m2m) {
  #pragma HLS INLINE
  return (k0 * kOuterTileSize + k1) * kSizeMMemory +
         (m0 * kOuterTileSizeMemory + m1 * kInnerTileSizeMMemory + m2m);
}

int IndexC(int n0, int n1, int n2, int m0, int m1, int m2m) {
  #pragma HLS INLINE
  return (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeMMemory +
         (m0 * kOuterTileSizeMemory + m1 * kInnerTileSizeMMemory + m2m);
}

int IndexABuffer(int n1, int n2, int k1) {
  #pragma HLS INLINE
  return k1 * kOuterTileSize + (n1 * kInnerTileSizeN + n2);
}

int IndexBBuffer(int k1, int m1, int m2m, int m2k) {
  #pragma HLS INLINE
  return k1 * kOuterTileSize + m1 * kInnerTileSizeM + m2m * kMemoryWidth + m2k;
}

int IndexBBuffer(int k1, int m1, int p2) {
  #pragma HLS INLINE
  return k1 * kOuterTileSize + m1 * kInnerTileSizeM + p2;
}

int IndexCBuffer(int n1, int n2, int m1, int m2m, int m2k) {
  #pragma HLS INLINE
  return (n1 * kInnerTileSizeN + n2) * kOuterTileSize +
         (m1 * kInnerTileSizeM + m2m * kMemoryWidth + m2k);
}

void ReadB(MemoryPack_t const b[], Stream<MemoryPack_t> &pipe) {
ReadB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  ReadB_OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
    ReadB_OuterTile_K:
      for (int k0 = 0; k0 < kOuterTilesK; ++k0) {
      ReadB_BufferB_K1:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        ReadB_BufferB_M1:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
          ReadB_BufferB_M2:
            for (int m2m = 0; m2m < kInnerTileSizeMMemory; ++m2m) {
              #pragma HLS LOOP_FLATTEN
              #pragma HLS PIPELINE II=1
              pipe.Push(b[IndexB(k0, k1, m0, m1, m2m)]);
            }
          }
        }
      }
    }
  }
}

void ComputeKernel(Data_t const a[], Stream<MemoryPack_t> &bPipe,
                   MemoryPack_t c[]) {
OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  OuterTile_M:
    for (int m0 = 0; m0 < kOuterTilesM; ++m0) {

      // Size equivalent to kOuterTileSize * kOuterTileSize
      Data_t cBuffer[kInnerTilesN * kInnerTilesM * kInnerTileSizeN * kInnerTileSizeM];
      #pragma HLS ARRAY_PARTITION variable=cBuffer cyclic factor=kInnerTileSizeN*kInnerTileSizeM
  
    OuterTile_K:
      for (int k0 = 0; k0 < kOuterTilesK; ++k0) {
        // Begin outer tile ---------------------------------------------------

        Data_t aBuffer[kOuterTileSize * kOuterTileSize];
        #pragma HLS ARRAY_PARTITION variable=aBuffer cyclic factor=kInnerTileSizeN
      BufferA_N1:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        BufferA_N2:
          for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
          BufferA_K1:
            for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              aBuffer[IndexABuffer(n1, n2, k1)] = a[IndexA(n0, n1, n2, k0, k1)];
            }
          }
        }

        Data_t bBuffer[kOuterTileSize * kOuterTileSize];
        #pragma HLS ARRAY_PARTITION variable=bBuffer cyclic factor=kInnerTileSizeM
      BufferB_K1:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        BufferB_M1:
          for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
          BufferB_M2M:
            for (int m2m = 0; m2m < kInnerTileSizeMMemory; ++m2m) {
              MemoryPack_t pack;
            BufferB_M2K:
              for (int m2k = 0; m2k < kMemoryWidth; ++m2k) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                if (m2k == 0) {
                  pack = bPipe.Pop();
                }
                bBuffer[IndexBBuffer(k1, m1, m2m, m2k)] = pack[m2k]; 
              }
            }
          }
        }

        // We do not tile M further, but loop over the entire outer tile here
      Pipeline_K:
        for (int k1 = 0; k1 < kOuterTileSize; ++k1) {
        
        Pipeline_N:
          for (int n1 = 0; n1 < kInnerTilesN; ++n1) {

          Pipeline_M:
            for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
              // Begin inner tile ---------------------------------------------
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
            
            Unroll_N:
              for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
                #pragma HLS UNROLL

                const auto aVal = aBuffer[IndexABuffer(n1, n2, k1)];

              Unroll_MM:
                for (int m2m = 0; m2m < kInnerTileSizeMMemory; ++m2m) {
                  #pragma HLS UNROLL

                Unroll_MK:
                  for (int m2k = 0; m2k < kMemoryWidth; ++m2k) {
                    #pragma HLS UNROLL
                    // Begin compute tile --------------------------------------

                    const auto bVal = bBuffer[IndexBBuffer(k1, m1, m2m, m2k)];

                    const auto mult = aVal * bVal;

                    const auto prev =
                        (k0 == 0 && k1 == 0)
                            ? 0
                            : cBuffer[IndexCBuffer(n1, n2, m1, m2m, m2k)];
                    cBuffer[IndexCBuffer(n1, n2, m1, m2m, m2k)] = prev + mult;
                    #pragma HLS DEPENDENCE variable=cBuffer false

                    // End compute tile ----------------------------------------
                  }
                }
              }
            }

            // End inner tile ---------------------------------------------------
          }
        }


        // End outer tile -----------------------------------------------------
      }

      // Write back this tile of C ---------------------------------------------
    WriteC_N1:
      for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
      WriteC_M1:
        for (int m1 = 0; m1 < kInnerTilesM; ++m1) {
        WriteC_N2:
          for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
          WriteC_M2M:
            for (int m2m = 0; m2m < kInnerTileSizeMMemory; ++m2m) {
              MemoryPack_t pack;
            WriteC_M2K:
              for (int m2k = 0; m2k < kMemoryWidth; ++m2k) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                pack[m2k] = cBuffer[IndexCBuffer(n1, n2, m1, m2m, m2k)];
                if (m2k == kMemoryWidth - 1) {
                  c[IndexC(n0, n1, n2, m0, m1, m2m)] = pack;
                }
              }
            }
          }
        }
      }
      // -----------------------------------------------------------------------

    }
  }

}

void MatrixMatrix(Data_t const a[], MemoryPack_t const b[],
                  MemoryPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmek0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmek1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  Stream<MemoryPack_t> bPipe;

  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bPipe);
  HLSLIB_DATAFLOW_FUNCTION(ComputeKernel, a, bPipe, c);
  HLSLIB_DATAFLOW_FINALIZE();
}
