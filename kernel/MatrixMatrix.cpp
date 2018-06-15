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

int IndexABuffer(int n1, int n2) {
  #pragma HLS INLINE
  return n1 * kInnerTileSize + n2;
}

int IndexBBuffer(int p1, int p2) {
  #pragma HLS INLINE
  return p1 * kInnerTileSize + p2;
}

int IndexCBuffer(int n1, int n2, int p1, int p2) {
  #pragma HLS INLINE
  return (n1 * kInnerTiles + p1) * kInnerTileSize * kInnerTileSize +
         n2 * kInnerTileSize + p2;
}

void ComputeKernel(Data_t const a[], Data_t const b[], Data_t c[]) {

  for (int n0 = 0; n0 < kOuterTilesN; ++n0) {
    for (int p0 = 0; p0 < kOuterTilesP; ++p0) {

      constexpr int kInnerTileSizeSquared = kInnerTileSize * kInnerTileSize;
      Data_t cBuffer[kInnerTiles * kInnerTiles * kInnerTileSizeSquared];
      #pragma HLS ARRAY_PARTITION variable=cBuffer cyclic factor=kInnerTileSizeSquared

      for (int m0 = 0; m0 < kOuterTilesM; ++m0) {
        // Begin outer tile ---------------------------------------------------

        // We do not tile M further, but loop over the entire outer tile here
        for (int m1 = 0; m1 < kOuterTileSize; ++m1) {

          Data_t aBuffer[kOuterTileSize];
          for (int n1 = 0; n1 < kInnerTiles; ++n1) {
            for (int n2 = 0; n2 < kInnerTileSize; ++n2) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              aBuffer[IndexABuffer(n1, n2)] = a[IndexA(n0, n1, n2, m0, m1)];
            }
          }

          Data_t bBuffer[kOuterTileSize];
          for (int p1 = 0; p1 < kInnerTiles; ++p1) {
            for (int p2 = 0; p2 < kInnerTileSize; ++p2) {
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
              bBuffer[IndexBBuffer(p1, p2)] = b[IndexB(m0, m1, p0, p1, p2)]; 
            }
          }
        
          for (int n1 = 0; n1 < kInnerTiles; ++n1) {

            for (int p1 = 0; p1 < kInnerTiles; ++p1) {
              // Begin inner tile ---------------------------------------------
              #pragma HLS PIPELINE II=1
              #pragma HLS LOOP_FLATTEN
            
              for (int n2 = 0; n2 < kInnerTileSize; ++n2) {
                #pragma HLS UNROLL

                const auto aVal = aBuffer[IndexABuffer(n1, n2)];

                for (int p2 = 0; p2 < kInnerTileSize; ++p2) {
                  #pragma HLS UNROLL
                  // Begin compute tile ---------------------------------------

                  const auto bVal = bBuffer[IndexBBuffer(p1, p2)];

                  const auto mult = aVal * bVal;

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

  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ComputeKernel, a, b, c);
  HLSLIB_DATAFLOW_FINALIZE();
}
