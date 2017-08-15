/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"

inline int GlobalIndex(int bn, int bp, int tn, int tp) {
  #pragma HLS INLINE
  return (bn * kTileSizeN + tn) * kSize + bp * kTileSizeP + tp;
}

inline int GlobalIndexKernel(int bn, int bp, int tn, int tp) {
  #pragma HLS INLINE
  return (bn * kTileSizeN + tn) * kSizeKernel + bp * kTileSizePKernel + tp;
}

void MatrixMatrix(Data_t const a[], KernelPack_t const b[], KernelPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

BlocksN:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  BlocksP:
    for (int bp = 0; bp < kBlocksP; ++bp) {

      KernelPack_t cLocal[kTileSizeN][kTileSizePKernel] = {Data_t(0)};
      #pragma HLS ARRAY_PARTITION variable=cLocal dim=1 complete

      // Loop over collapsed dimension
    CollapseM:
      for (int m = 0; m < kSize; ++m) {

        // Load column of A
        Data_t aColumn[kTileSizeN]; // Local buffer of A
      LoadA:
        for (int tn = 0; tn < kTileSizeN; ++tn) {
          #pragma HLS PIPELINE
          aColumn[tn] = a[GlobalIndex(bn, 0, tn, m)]; 
        }

        // Stream row of B and apply it to all correspondings elements of A 
      StreamB:
        for (int tp = 0; tp < kTileSizePKernel; ++tp) {
          #pragma HLS PIPELINE
          const auto readB = b[GlobalIndexKernel(0, bp, m, tp)];
        UnrollA:
          for (int tn = 0; tn < kTileSizeN; ++tn) {
            #pragma HLS UNROLL
            const auto aVal = aColumn[tn]; 
            const auto cAcc = cLocal[tn][tp];
            KernelPack_t result;
          UnrollVector:
            for (int w = 0; w < kKernelWidth; ++w) {
              #pragma HLS UNROLL
              const auto map = OperatorMap::Apply(readB[w], aVal);
              result[w] = OperatorReduce::Apply(map, cAcc[w]);
            }
            cLocal[tn][tp] = result;
          }
        }

      }

      // Write out result block
    StoreCRows:
      for (int tn = 0; tn < kTileSizeN; ++tn) {
      StoreCCols:
        for (int tp = 0; tp < kTileSizePKernel; ++tp) {
          #pragma HLS PIPELINE
          c[GlobalIndexKernel(bn, bp, tn, tp)] = cLocal[tn][tp];
        }
      }
    }
  }
}
