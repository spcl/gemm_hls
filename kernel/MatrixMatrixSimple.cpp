#include "MatrixMatrix.h"
#include "Memory.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"
#include "hlslib/DataPack.h"

void MatrixMatrixSimple(hlslib::Stream<Data_t> &aPipe,
                        hlslib::Stream<KernelPack_t> &bPipe,
                        hlslib::Stream<KernelPack_t> &cPipe) {
Outer_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  Outer_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {

      KernelPack_t acc[kTileSizeN][kTileSizePKernel];
      #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

    Loop_M:
      for (int m = 0; m < kSizeM; ++m) {

        Data_t a_buffer[kTileSizeN];
      Inner_N:
        for (int nu = 0; nu < kTileSizeN; ++nu) {
          #pragma HLS PIPELINE
          a_buffer[nu] = hlslib::ReadBlocking(aPipe);
        }

      Inner_P:
        for (int p = 0; p < kTileSizePKernel; ++p) {
          #pragma HLS PIPELINE
          const auto b_val = hlslib::ReadBlocking(bPipe);
        Unroll_N:
          for (int nu = 0; nu < kTileSizeN; ++nu) {
            #pragma HLS UNROLL
            const auto a_val = a_buffer[nu];
            const auto prev =
                (m > 0) ? acc[nu][p] : KernelPack_t(OperatorReduce::identity());
            KernelPack_t updated;
            for (int w = 0; w < kKernelWidth; ++w) {
              #pragma HLS UNROLL
              const auto mapped = OperatorMap::Apply(b_val[w], a_val);
              updated[w] = OperatorReduce::Apply(prev[w], mapped); 
            }
            acc[nu][p] = updated; 
            #pragma HLS DEPENDENCE variable=acc inter false
          }
        }

      } // End M-loop

    WriteC_N:
      for (int nu = 0; nu < kTileSizeN; ++nu) {
      WriteC_P:
        for (int p = 0; p < kTileSizePKernel; ++p) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          hlslib::WriteBlocking(cPipe, acc[nu][p]);
        }
      }

    } // End loop over P-tiles
  } // End loop over N-tiles

}

void MatrixMatrix(MemoryPack_t const a[], MemoryPack_t const b[],
                  MemoryPack_t c[]) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=c bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  hlslib::Stream<MemoryPack_t> aMem("aMem");
#ifndef MM_SYNTHESIS
  hlslib::Stream<Data_t> aSplit[kMemoryWidth];
#else
  hls::stream<Data_t> aSplit[kMemoryWidth];
  #pragma HLS STREAM variable=aSplit depth=kTransposeDepth
#endif
  hlslib::Stream<Data_t> aKernel("aKernel");
  hlslib::Stream<MemoryPack_t> bMem("bMem");
  hlslib::Stream<KernelPack_t> bKernel("bMem");
  hlslib::Stream<KernelPack_t> cKernel("cKernel");
  hlslib::Stream<MemoryPack_t> cMem("cMem");

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadASplit, a, aSplit);
  HLSLIB_DATAFLOW_FUNCTION(ReadARotate, aSplit, aKernel);
  HLSLIB_DATAFLOW_FUNCTION(ReadBMemory, b, bMem);
  HLSLIB_DATAFLOW_FUNCTION(ReadBKernel, bMem, bKernel);

  HLSLIB_DATAFLOW_FUNCTION(MatrixMatrixSimple, aKernel, bKernel, cKernel); 

  HLSLIB_DATAFLOW_FUNCTION(WriteCKernel, cKernel, cMem);
  HLSLIB_DATAFLOW_FUNCTION(WriteCMemory, cMem, c);

  HLSLIB_DATAFLOW_FINALIZE();
}
