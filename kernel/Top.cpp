#include "MatrixMultiplication.h"
#include "Compute.h"
#include "Memory.h"
#include "hlslib/xilinx/Simulation.h"

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
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aConvert, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthA, aConvert, aPipes[0], size_n, size_k,
                           size_m);
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
