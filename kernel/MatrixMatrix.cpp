/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
// #include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

using hlslib::Stream;
using hlslib::DataPack;

int IndexA(int n0, int n1, int n2, int m0, int m1) {
  #pragma HLS INLINE
  int index =  (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeK +
         (m0 * kOuterTileSize + m1);
  assert(index < kSizeN * kSizeK);
  return (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeK +
         (m0 * kOuterTileSize + m1);
}

int IndexB(int m0, int m1, int p0, int p1, int p2m) {
  #pragma HLS INLINE
  int index =  (m0 * kOuterTileSize + m1) * kSizeMMemory +
         (p0 * kOuterTileSizeMemory + p1 * kInnerTileSizeMMemory + p2m);
  assert(index < kSizeK * kSizeM);
  return (m0 * kOuterTileSize + m1) * kSizeMMemory +
         (p0 * kOuterTileSizeMemory + p1 * kInnerTileSizeMMemory + p2m);
}

int IndexC(int n0, int n1, int n2, int p0, int p1, int p2m) {
  #pragma HLS INLINE
  int index =  (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeMMemory +
         (p0 * kOuterTileSizeMemory + p1 * kInnerTileSizeMMemory + p2m);
  assert(index < kSizeN * kSizeM);
  return (n0 * kOuterTileSize + n1 * kInnerTileSizeN + n2) * kSizeMMemory +
         (p0 * kOuterTileSizeMemory + p1 * kInnerTileSizeMMemory + p2m);
}

int IndexABuffer(int n1, int n2, int m1) {
  #pragma HLS INLINE
  int index =  m1 * kOuterTileSize + (n1 * kInnerTileSizeN + n2);
  assert(index < kOuterTileSize * kOuterTileSize);
  return m1 * kOuterTileSize + (n1 * kInnerTileSizeN + n2);
}

int IndexBBuffer(int m1, int p1, int p2m, int p2k) {
  #pragma HLS INLINE
  int index =  m1 * kOuterTileSize + p1 * kInnerTileSizeM + p2m * kMemoryWidth + p2k;
  assert(index < kOuterTileSize * kOuterTileSize);
  return m1 * kOuterTileSize + p1 * kInnerTileSizeM + p2m * kMemoryWidth + p2k;
}

int IndexBBuffer(int m1, int p1, int p2) {
  #pragma HLS INLINE
  int index =  m1 * kOuterTileSize + p1 * kInnerTileSizeM + p2;
  assert(index < kOuterTileSize * kOuterTileSize);
  return m1 * kOuterTileSize + p1 * kInnerTileSizeM + p2;
}

int IndexCBuffer(int n1, int n2, int p1, int p2m, int p2k) {
  #pragma HLS INLINE
  return (n1 * kInnerTileSizeN + n2) * kOuterTileSize +
         (p1 * kInnerTileSizeM + p2m * kMemoryWidth + p2k);
}

void ReadB(MemoryPack_t const b[], Stream<MemoryPack_t> &pipe) {
ReadB_OuterTile_N:
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
  ReadB_OuterTile_P:
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {
    ReadB_OuterTile_M:
      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
      ReadB_BufferB_M1:
        for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
        ReadB_BufferB_P1:
          for (int p1 = 0; p1 < kInnerTilesM; ++p1) {
          ReadB_BufferB_P2:
            for (int p2m = 0; p2m < kInnerTileSizeMMemory; ++p2m) {
              #pragma HLS LOOP_FLATTEN
              #pragma HLS PIPELINE II=1
              pipe.Push(b[IndexB(m0, m1, p0, p1, p2m)]);
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
  OuterTile_P:
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {

      // Size equivalent to kOuterTileSize * kOuterTileSize
      Data_t cBuffer[kInnerTilesN * kInnerTilesM * kInnerTileSizeN * kInnerTileSizeM];
      #pragma HLS ARRAY_PARTITION variable=cBuffer cyclic factor=kInnerTileSizeN*kInnerTileSizeM
  
    OuterTile_M:
      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
        // Begin outer tile ---------------------------------------------------

        Data_t aBuffer[kOuterTileSize * kOuterTileSize];
        #pragma HLS ARRAY_PARTITION variable=aBuffer cyclic factor=kInnerTileSizeN
      BufferA_N1:
        for (int n1 = 0; n1 < kInnerTilesN; ++n1) {
        BufferA_N2:
          for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
          BufferA_M1:
            for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              aBuffer[IndexABuffer(n1, n2, m1)] = a[IndexA(n0, n1, n2, m0, m1)];
            }
          }
        }

        Data_t bBuffer[kOuterTileSize * kOuterTileSize];
        #pragma HLS ARRAY_PARTITION variable=bBuffer cyclic factor=kInnerTileSizeM
      BufferB_M1:
        for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
        BufferB_P1:
          for (int p1 = 0; p1 < kInnerTilesM; ++p1) {
          BufferB_P2M:
            for (int p2m = 0; p2m < kInnerTileSizeMMemory; ++p2m) {
              MemoryPack_t pack;
            BufferB_P2K:
              for (int p2k = 0; p2k < kMemoryWidth; ++p2k) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                if (p2k == 0) {
                  pack = bPipe.Pop();
                }
                bBuffer[IndexBBuffer(m1, p1, p2m, p2k)] = pack[p2k]; 
              }
            }
          }
        }

        // We do not tile M further, but loop over the entire outer tile here
      Pipeline_M:
        for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
        
        Pipeline_N:
          for (int n1 = 0; n1 < kInnerTilesN; ++n1) {

          Pipeline_P:
            for (int p1 = 0; p1 < kInnerTilesM; ++p1) {
              // Begin inner tile ---------------------------------------------
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
            
            Unroll_N:
              for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
                #pragma HLS UNROLL

                const auto aVal = aBuffer[IndexABuffer(n1, n2, m1)];

              Unroll_PM:
                for (int p2m = 0; p2m < kInnerTileSizeMMemory; ++p2m) {
                  #pragma HLS UNROLL

                Unroll_PK:
                  for (int p2k = 0; p2k < kMemoryWidth; ++p2k) {
                    #pragma HLS UNROLL
                    // Begin compute tile --------------------------------------

                    const auto bVal = bBuffer[IndexBBuffer(m1, p1, p2m, p2k)];

                    const auto mult = aVal * bVal;

                    const auto prev =
                        (m0 == 0 && m1 == 0)
                            ? 0
                            : cBuffer[IndexCBuffer(n1, n2, p1, p2m, p2k)];
                    cBuffer[IndexCBuffer(n1, n2, p1, p2m, p2k)] = prev + mult;
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
      WriteC_P1:
        for (int p1 = 0; p1 < kInnerTilesM; ++p1) {
        WriteC_N2:
          for (int n2 = 0; n2 < kInnerTileSizeN; ++n2) {
          WriteC_P2M:
            for (int p2m = 0; p2m < kInnerTileSizeMMemory; ++p2m) {
              MemoryPack_t pack;
            WriteC_P2K:
              for (int p2k = 0; p2k < kMemoryWidth; ++p2k) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                pack[p2k] = cBuffer[IndexCBuffer(n1, n2, p1, p2m, p2k)];
                if (p2k == kMemoryWidth - 1) {
                  c[IndexC(n0, n1, n2, p0, p1, p2m)] = pack;
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

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
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
