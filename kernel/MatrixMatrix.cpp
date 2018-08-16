/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "MatrixMatrix.h"
#include "Memory.h"
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

// -------------    --------------   --------------
// | A  A  A  A |   | B  B  .  . |   | C  C  .  . |
// | A  A  A  A |   | B  B  .  . |   | C  C  .  . |
// | .  .  .  . | X | B  B  .  . | = | .  .  .  . |
// | .  .  .  . |   | B  B  .  . |   | .  .  .  . |
// | .  .  .  . |   | B  B  .  . |   | .  .  .  . |
// -------------    --------------   --------------

void MatrixMatrixStage(int id, hlslib::Stream<Data_t> &aIn,
                       hlslib::Stream<KernelPack_t> &bIn,
                       hlslib::Stream<KernelPack_t> &cIn,
                       hlslib::Stream<Data_t> &aOut,
                       hlslib::Stream<KernelPack_t> &bOut,
                       hlslib::Stream<KernelPack_t> &cOut) {
  
  // Manual indices for flattened loop
  int i_loadA_tn = 0;
  const int i_loadA_tn_end = kTileSizeN - id;
  int i_streamB_tp = 0;
  const int i_streamB_tp_end = kTileSizePKernel;
  int i_outer = 0;
  const int i_outer_end = kSizeM;
  int i_storeC = 0;
  const int i_storeC_end = (id + 1) * kTileSizePKernel;
  const int i_saturated_end = kTileSizeN - id; 

  //============================================================================
  // Outer loops over tiles in rows (N) and columns (P) of C, respectively 
  //============================================================================

Blocks_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  Blocks_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {

      KernelPack_t cLocal[kTileSizePKernel]; // Partial results buffer

      Data_t aNext, aVal; // Shift registers for forwarding A

      // Manually flattened loop
      const int loopBound =
          i_loadA_tn_end + kSizeM * i_streamB_tp_end + i_storeC_end;

      //========================================================================
      // Inner, pipelined loop over each tile 
      //========================================================================
    Flattened:
      for (int i = 0; i < loopBound; ++i) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE

        if (i < loopBound - i_storeC_end) {

          // Grab next from previous iteration. This way we avoid that the
          // last processing elements overwrites its next value before it is
          // used
          if (i_streamB_tp == 0) {
            aVal = aNext;
          }
          const bool loadA =
              (i_streamB_tp < i_loadA_tn_end) && (i_outer < i_outer_end - 1);
          if (loadA) {
            const auto aRead = aIn.Pop();
            // Don't forward on the last iteration
            if (i_loadA_tn < kTileSizeN - id - 1) {
              if (id < kTileSizeN - 1) { // Otherwise Vivado HLS cannot infer 
                aOut.Push(aRead);        // that this is never written
              }
            } else {
              aNext = aRead;
            }
            if (i_loadA_tn == i_loadA_tn_end - 1) {
              i_loadA_tn = 0;
            } else {
              ++i_loadA_tn;
            }
          }
          if (i < i_saturated_end) {
            continue; // Buffer first column of A before streaming
          }
          const auto readB = bIn.Pop(); 
          if (id < kTileSizeN - 1) {
            bOut.Push(readB); // Forward B to next PE 
          }
          KernelPack_t cAcc;
          if (i_outer > 0) {
            cAcc = cLocal[i_streamB_tp];
            #pragma HLS DEPENDENCE variable=cLocal inter false
          } else {
            cAcc = KernelPack_t(OperatorReduce::identity());
          }
          KernelPack_t result;
        UnrollVector:
          for (int w = 0; w < kKernelWidth; ++w) {
            #pragma HLS UNROLL
            const auto mapped = OperatorMap::Apply(readB[w], aVal);
            result[w] = OperatorReduce::Apply(mapped, cAcc[w]);
          }
          cLocal[i_streamB_tp] = result;
          #pragma HLS DEPENDENCE variable=cLocal inter false

          // Manual index calculation
          if (i_streamB_tp == i_streamB_tp_end - 1) {
            i_streamB_tp = 0;
            if (i_outer == i_outer_end - 1) {
              i_outer = 0;
            } else {
              ++i_outer;
            }
          } else {
            ++i_streamB_tp;
          }

        } else {

          //====================================================================
          // Write back C in separate iterations 
          //====================================================================

          if (i_storeC < kTileSizePKernel) {
            cOut.Push(cLocal[i_storeC]);
            #pragma HLS DEPENDENCE variable=cLocal inter false
          } else {
            // Vivado HLS cannot deduce that this never happens for id == 0, so
            // put an explicit condition here
            if (id > 0) {
              cOut.Push(cIn.Pop());
            }
          }
          if (i_storeC == i_storeC_end - 1) {
            i_storeC = 0;
          } else {
            ++i_storeC;
          }

        }

      }

    }
  }
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

  //----------------------------------------------------------------------------
  // Module interconnect 
  //----------------------------------------------------------------------------

  // Memory/kernel width conversion pipes
  hlslib::Stream<MemoryPack_t> aMem("aMem");
  hlslib::Stream<MemoryPack_t> bMem("bMem");
  hlslib::Stream<MemoryPack_t> cMem("cMem");
  // Transposition pipes for reading A in bursts
  hlslib::Stream<Data_t, kTransposeDepth> aSplit[kMemoryWidth];
  #pragma HLS STREAM variable=aSplit depth=kTransposeDepth
  // Systolic array pipes for A, B and C
  hlslib::Stream<Data_t> aPipes[kTileSizeN + 1];
  hlslib::Stream<KernelPack_t> bPipes[kTileSizeN + 1];
  hlslib::Stream<KernelPack_t> cPipes[kTileSizeN + 1];

  HLSLIB_DATAFLOW_INIT();

  //============================================================================
  // Memory modules 
  //============================================================================
  
  HLSLIB_DATAFLOW_FUNCTION(ReadASplit, a, aSplit);
  HLSLIB_DATAFLOW_FUNCTION(ReadARotate, aSplit, aPipes[0]);
  HLSLIB_DATAFLOW_FUNCTION(ReadBMemory, b, bMem);
  HLSLIB_DATAFLOW_FUNCTION(ReadBKernel, bMem, bPipes[0]);
  HLSLIB_DATAFLOW_FUNCTION(WriteCKernel, cPipes[kTileSizeN], cMem);
  HLSLIB_DATAFLOW_FUNCTION(WriteCMemory, cMem, c);

  //============================================================================
  // Name pipes for debugging simulation
  //============================================================================

#ifndef MM_SYNTHESIS
  for (int mw = 0; mw < kMemoryWidth; ++mw) {
    aSplit[mw].set_name(("aSplit[" + std::to_string(mw) + "]").c_str());
  }
  int arr[kTileSizeN];
  for (int tn = 0; tn < kTileSizeN; ++tn) {
    aPipes[tn].set_name(("aPipes[" + std::to_string(tn) + "]").c_str());
    bPipes[tn].set_name(("bPipes[" + std::to_string(tn) + "]").c_str());
    cPipes[tn].set_name(("cPipes[" + std::to_string(tn) + "]").c_str());
  }
  aPipes[kTileSizeN].set_name(
      ("aPipes[" + std::to_string(kTileSizeN) + "]").c_str());
  bPipes[kTileSizeN].set_name(
      ("bPipes[" + std::to_string(kTileSizeN) + "]").c_str());
  cPipes[kTileSizeN].set_name(
      ("cPipes[" + std::to_string(kTileSizeN) + "]").c_str());
#endif

  //============================================================================
  // Compute modules in systolic array 
  //============================================================================

UnrollCompute:
  for (int tn = 0; tn < kTileSizeN; ++tn) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(MatrixMatrixStage, tn, aPipes[tn], bPipes[tn],
                             cPipes[tn], aPipes[tn + 1], bPipes[tn + 1],
                             cPipes[tn + 1]);
  }

  HLSLIB_DATAFLOW_FINALIZE();
}
