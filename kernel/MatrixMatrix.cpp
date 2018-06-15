/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
#include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

using hlslib::Stream;
using hlslib::DataPack;

int IndexA(int n0, int n1, int n2, int m0, int m1) {
  #pragma HLS INLINE
  return (n0 * kOuterTileSize + n1 * kInnerTileSize + n2) * kSizeM +
         (m0 * kOuterTileSize + m1);
}

int IndexB(int m0, int m1, int p0, int p1, int p2) {
  #pragma HLS INLINE
  return (m0 * kOuterTileSize + m1) * kSizeP +
         (p0 * kOuterTileSize + p1 * kInnerTileSize + p2);
}

int IndexC(int n0, int n1, int n2, int p0, int p1, int p2) {
  #pragma HLS INLINE
  return (n0 * kOuterTileSize + n1 * kInnerTileSize + n2) * kSizeP +
         (p0 * kOuterTileSize + p1 * kInnerTileSize + p2);
}

int IndexCBuffer(int n1, int n2, int p1, int p2) {
  #pragma HLS INLINE
  return (n1 * kInnerTiles + p1) * kInnerTileSize * kInnerTileSize +
         n2 * kInnerTileSize + p2;
}

void ReadA(Data_t const a[], Stream<DataPack<Data_t, kInnerTileSize>> &pipe) {
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {
      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
        for (int n1 = 0; n1 < kInnerTiles; ++n1) {
          for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
            for (int p1 = 0; p1 < kInnerTiles; ++p1) {
              DataPack<Data_t, kInnerTileSize> aPack;
              for (int n2 = 0; n2 < kInnerTileSize; ++n2) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                aPack[n2] = a[IndexA(n0, n1, n2, m0, m1)];
                if (n2 == kInnerTileSize - 1) {
                  pipe.Push(aPack);
                }
              }
            }
          }
        }
      }
    }
  }
}

void ReadB(Data_t const b[], Stream<DataPack<Data_t, kInnerTileSize>> &pipe) {
  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {
      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
        for (int n1 = 0; n1 < kInnerTiles; ++n1) {
          for (int m1 = 0; m1 < kOuterTileSize; ++m1) {
            for (int p1 = 0; p1 < kInnerTiles; ++p1) {
              DataPack<Data_t, kInnerTileSize> bPack;
              for (int p2 = 0; p2 < kInnerTileSize; ++p2) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                bPack[p2] = b[IndexB(m0, m1, p0, p1, p2)];
                if (p2 == kInnerTileSize - 1) {
                  pipe.Push(bPack);
                }
              }
            }
          }
        }
      }
    }
  }
}

void ComputeKernel(Stream<DataPack<Data_t, kInnerTileSize>> &aPipe,
                   Stream<DataPack<Data_t, kInnerTileSize>> &bPipe,
                   Data_t c[]) {

  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {

      constexpr int kInnerTileSizeSquared = kInnerTileSize * kInnerTileSize;
      Data_t cBuffer[kInnerTiles * kInnerTiles * kInnerTileSizeSquared];
      #pragma HLS ARRAY_PARTITION variable=cBuffer cyclic factor=kInnerTileSizeSquared

      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
        // Begin outer tile ---------------------------------------------------
        
        for (int n1 = 0; n1 < kInnerTiles; ++n1) {

          // We do not tile M further, but loop over the entire outer tile here
          for (int m1 = 0; m1 < kOuterTileSize; ++m1) {

            for (int p1 = 0; p1 < kInnerTiles; ++p1) {
              // Begin inner tile ---------------------------------------------
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN

              const auto aPack = aPipe.Pop();
              const auto bPack = bPipe.Pop();
            
              for (int n2 = 0; n2 < kInnerTileSize; ++n2) {
                #pragma HLS UNROLL
                for (int p2 = 0; p2 < kInnerTileSize; ++p2) {
                  #pragma HLS UNROLL
                  // Begin compute tile ---------------------------------------

                  const auto mult = aPack[n2] * bPack[p2];

                  const auto prev = (m0 == 0 && m1 == 0)
                                        ? 0
                                        : cBuffer[IndexCBuffer(n1, n2, p1, p2)];
                  cBuffer[IndexCBuffer(n1, n2, p1, p2)] = prev + mult;
                  #pragma HLS DEPENDENCE variable=cBuffer false

                  // End compute tile -----------------------------------------
                }
              }
            }

            // End inner tile ---------------------------------------------------
          }
        }


        // End outer tile -----------------------------------------------------
      }

      // Write back this tile of C ---------------------------------------------
      for (int n1 = 0; n1 < kInnerTiles; ++n1) {
        for (int p1 = 0; p1 < kInnerTiles; ++p1) {
          for (int n2 = 0; n2 < kInnerTileSize; ++n2) {
            for (int p2 = 0; p2 < kInnerTileSize; ++p2) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              c[IndexC(n0, n1, n2, p0, p1, p2)] =
                  cBuffer[IndexCBuffer(n1, n2, p1, p2)];
            }
          }
        }
      }
      // -----------------------------------------------------------------------

    }
  }

}

void MatrixMatrix(Data_t const a[], Data_t const b[],
                  Data_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  Stream<DataPack<Data_t, kInnerTileSize>> aPipe("aPipe");
  Stream<DataPack<Data_t, kInnerTileSize>> bPipe("bPipe");

  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aPipe);
  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bPipe);
  HLSLIB_DATAFLOW_FUNCTION(ComputeKernel, aPipe, bPipe, c);
  HLSLIB_DATAFLOW_FINALIZE();
}
