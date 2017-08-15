/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include <sstream>
#include <iostream>

inline int GlobalIndex(int bn, int bp, int tn, int tp) {
  #pragma HLS INLINE
  return (bn * kTileSizeN + tn) * kSize + bp * kTileSizeP + tp;
}

inline int GlobalIndexKernel(int bn, int bp, int tn, int tp) {
  #pragma HLS INLINE
  return (bn * kTileSizeN + tn) * kSizeKernel + bp * kTileSizePKernel + tp;
}

enum class State {
  loadingA,
  streamingB,
  storingC
};

void MatrixMatrixStage(int id,
                       hlslib::Stream<Data_t> &aIn,
                       hlslib::Stream<KernelPack_t> &bIn,
                       hlslib::Stream<KernelPack_t> &cIn,
                       hlslib::Stream<Data_t> &aOut,
                       hlslib::Stream<KernelPack_t> &bOut,
                       hlslib::Stream<KernelPack_t> &cOut) {

Blocks_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  Blocks_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {

// #ifndef MM_SYNTHESIS
//       hlslib::Stream<KernelPack_t> cLocal("cLocal");
// #else
//       hls::stream<KernelPack_t> cLocal("cLocal");
//       #pragma HLS STREAM variable=cLocal depth=kTileSizePKernel
// #endif
      KernelPack_t cLocal[kTileSizePKernel];
      State state = State::loadingA;

      int i_loadA_tn = 0;
      const int i_loadA_tn_end = kTileSizeN - id;
      int i_streamB_tp = 0;
      const int i_streamB_tp_end = kTileSizePKernel;
      int i_outer = 0;
      const int i_outer_end = kSize;
      int i_storeC_tn = 0;
      const int i_storeC_tn_end = id + 1;
      int i_storeC_tp = 0;
      const int i_storeC_tp_end = kTileSizePKernel;
      Data_t aVal;

      // Manually flattened loop
    Flattened:
      for (int i = 0; i < kSize * (i_loadA_tn_end + i_streamB_tp_end) +
                              i_storeC_tn_end * i_storeC_tp_end;
           ++i) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE

        switch (state) {

          case State::loadingA: {
            aVal = hlslib::ReadBlocking(aIn);
            // Don't write on the last iteration
            if (i_loadA_tn < kTileSizeN - id - 1) {
              hlslib::WriteBlocking(aOut, aVal, 1);
            }
            if (i_loadA_tn == i_loadA_tn_end - 1) {
              i_loadA_tn = 0;
              state = State::streamingB;
            } else {
              ++i_loadA_tn;
            }
            break;
          }

          case State::streamingB: {
            const auto readB = hlslib::ReadBlocking(bIn); 
            if (id < kTileSizeN - 1) {
              hlslib::WriteBlocking(bOut, readB, 1); // Forward B
            }
            KernelPack_t cAcc;
            if (i_outer > 0) {
              cAcc = cLocal[i_streamB_tp];
              #pragma HLS DEPENDENCE variable=cLocal inter false
              // cAcc = hlslib::ReadOptimistic(cLocal);
            } else {
              cAcc = KernelPack_t(OperatorReduce::identity());
            }
            KernelPack_t result;
          UnrollVector:
            for (int w = 0; w < kKernelWidth; ++w) {
              #pragma HLS UNROLL
              const auto map = OperatorMap::Apply(readB[w], aVal);
              result[w] = OperatorReduce::Apply(map, cAcc[w]);
            }
            // hlslib::WriteOptimistic(cLocal, result, kTileSizePKernel);
            cLocal[i_streamB_tp] = result;
            #pragma HLS DEPENDENCE variable=cLocal inter false
            if (i_streamB_tp == i_streamB_tp_end - 1) {
              i_streamB_tp = 0;
              if (i_outer == i_outer_end - 1) {
                i_outer = 0;
                state = State::storingC;
              } else {
                ++i_outer;
                state = State::loadingA;
              }
            } else {
              ++i_streamB_tp;
            }
            break;
          }

          case State::storingC: {
            if (i_storeC_tn == 0) {
              // hlslib::WriteBlocking(cOut, hlslib::ReadOptimistic(cLocal), 1);
              hlslib::WriteBlocking(cOut, cLocal[i_storeC_tp], 1);
              #pragma HLS DEPENDENCE variable=cLocal inter false
            } else {
              hlslib::WriteBlocking(cOut, hlslib::ReadBlocking(cIn), 1);
            }
            if (i_storeC_tp == i_storeC_tp_end - 1) {
              i_storeC_tp = 0;
              if (i_storeC_tn == i_storeC_tn_end - 1) {
                i_storeC_tn = 0;
                state = State::loadingA;
              } else {
                ++i_storeC_tn;
              }
            } else {
              ++i_storeC_tp;
            }
            break;
          }

        }

      }
    }
  }
}

void ReadA(Data_t const a[], hlslib::Stream<Data_t> &aPipe) {

ReadA_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadA_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadA_M:
      for (int m = 0; m < kSize; ++m) {
      ReadA_N:
        for (int tn = 0; tn < kTileSizeN; ++tn) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          hlslib::WriteBlocking(aPipe, a[GlobalIndex(bn, 0, tn, m)], 1);
        }
      }
    }
  }
}

void ReadB(KernelPack_t const b[], hlslib::Stream<KernelPack_t> &bPipe) {
ReadB_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadB_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadB_M:
      for (int m = 0; m < kSize; ++m) {
      ReadB_P:
        for (int tp = 0; tp < kTileSizePKernel; ++tp) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          hlslib::WriteBlocking(bPipe, b[GlobalIndexKernel(0, bp, m, tp)], 1);
        }
      }
    }
  }
}

void WriteC(hlslib::Stream<KernelPack_t> &cPipe, KernelPack_t c[]) {
WriteC_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  WriteC_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    WriteC_N:
      for (int tn = 0; tn < kTileSizeN; ++tn) {
      WriteC_P:
        for (int tp = 0; tp < kTileSizePKernel; ++tp) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          c[GlobalIndexKernel(bn, bp, tn, tp)] = hlslib::ReadBlocking(cPipe);
        }
      }
    }
  }
}

void MatrixMatrix(Data_t const a[], KernelPack_t const b[], KernelPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  hlslib::Stream<Data_t> aPipes[kTileSizeN + 1];
  hlslib::Stream<KernelPack_t> bPipes[kTileSizeN + 1];
  hlslib::Stream<KernelPack_t> cPipes[kTileSizeN + 1];

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aPipes[0]);
  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bPipes[0]);

#ifdef MM_SYNTHESIS
  for (int tn = 0; tn < kTileSizeN; ++tn) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(MatrixMatrixStage, tn, aPipes[tn], bPipes[tn],
                             cPipes[tn], aPipes[tn + 1], bPipes[tn + 1],
                             cPipes[tn + 1]);
  }
#else
  int arr[kTileSizeN];
  for (int tn = 0; tn < kTileSizeN; ++tn) {
    #pragma HLS UNROLL
    arr[tn] = tn; // Need to allow passing by value
    aPipes[tn].set_name("aPipes[" + std::to_string(tn) + "]");
    bPipes[tn].set_name("bPipes[" + std::to_string(tn) + "]");
    cPipes[tn].set_name("cPipes[" + std::to_string(tn) + "]");
    HLSLIB_DATAFLOW_FUNCTION(MatrixMatrixStage, arr[tn], aPipes[tn], bPipes[tn],
                             cPipes[tn], aPipes[tn + 1], bPipes[tn + 1],
                             cPipes[tn + 1]);
  }
  aPipes[kTileSizeN].set_name("aPipes[" + std::to_string(kTileSizeN) + "]");
  bPipes[kTileSizeN].set_name("bPipes[" + std::to_string(kTileSizeN) + "]");
  cPipes[kTileSizeN].set_name("cPipes[" + std::to_string(kTileSizeN) + "]");
#endif

  HLSLIB_DATAFLOW_FUNCTION(WriteC, cPipes[kTileSizeN], c);

  HLSLIB_DATAFLOW_FINALIZE();
}
