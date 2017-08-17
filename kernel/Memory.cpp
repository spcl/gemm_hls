/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Memory.h"

inline int GetAIndex(int bn, int tn, int m) {
  #pragma HLS INLINE
  return bn * kTileSizeN * kSizeMMemory + tn * kSizeMMemory + m;
}

inline int GetBIndex(int bp, int m, int tp) {
  #pragma HLS INLINE
  return m * kSizePMemory + bp * kTileSizePMemory + tp;
}

inline int GetCIndex(int bn, int bp, int tn, int tp) {
  #pragma HLS INLINE
  return bn * kSizePMemory * kTileSizeN + tn * kSizePMemory +
         bp * kTileSizePMemory + tp;
}

void ReadBMemory(MemoryPack_t const b[], hlslib::Stream<MemoryPack_t> &bPipe) {
ReadBMemory_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadBMemory_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadBMemory_M:
      for (int m = 0; m < kSizeM; ++m) {
      ReadBMemory_P:
        for (int tp = 0; tp < kTileSizePMemory; ++tp) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          hlslib::WriteBlocking(bPipe, b[GetBIndex(bp, m, tp)], 1);
        }
      }
    }
  }

}

void ReadBKernel(hlslib::Stream<MemoryPack_t> &bMem,
                 hlslib::Stream<KernelPack_t> &bPipe) {
ReadB_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadB_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadB_M:
      for (int m = 0; m < kSizeM; ++m) {
      ReadB_P_Memory:
        for (int tpm = 0; tpm < kTileSizePMemory; ++tpm) {
          MemoryPack_t mem;
        ReadB_P_KernelPerMemory:
          for (int kpm = 0; kpm < kKernelPerMemory; ++kpm) { 
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE
            if (kpm == 0) {
              mem = hlslib::ReadBlocking(bMem);
            }
            const KernelPack_t kernel = mem[kpm];
            hlslib::WriteBlocking(bPipe, kernel, 1);
          }
        }
      }
    }
  }
}

void WriteCKernel(hlslib::Stream<KernelPack_t> &cPipe,
                  hlslib::Stream<MemoryPack_t> &cMem) {
WriteCKernel_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  WriteCKernel_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    WriteCKernel_N:
      for (int tn = 0; tn < kTileSizeN; ++tn) {
      WriteCKernel_P_Memory:
        for (int tp = 0; tp < kTileSizePMemory; ++tp) {
          MemoryPack_t mem;
        WriteCKernel_P_Kernel:
          for (int kpm = 0; kpm < kKernelPerMemory; ++kpm) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE
            mem[kpm] = hlslib::ReadBlocking(cPipe);
            if (kpm == kKernelPerMemory - 1) {
              hlslib::WriteBlocking(cMem, mem, 1);
            }
          }
        }
      }
    }
  }
}

void WriteCMemory(hlslib::Stream<MemoryPack_t> &cMem, MemoryPack_t c[]) {
WriteCMemory_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  WriteCMemory_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    WriteCMemory_N:
      for (int tn = 0; tn < kTileSizeN; ++tn) {
      WriteCMemory_P:
        for (int tp = 0; tp < kTileSizePMemory; ++tp) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          c[GetCIndex(bn, bp, tn, tp)] = hlslib::ReadBlocking(cMem);
        }
      }
    }
  }
}

// Split A into individual, deep FIFOs 
void ReadASplit(MemoryPack_t const a[],
                hlslib::Stream<Data_t> aSplit[kMemoryWidth]) {
ReadASplit_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadASplit_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadASplit_M:
      for (int m = 0; m < kSizeMMemory; ++m) {
      ReadASplit_N:
        for (int tn = 0; tn < kTileSizeN; ++tn) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE
          const auto read = a[GetAIndex(bn, tn, m)];
        ReadASplit_KernelPerMemory:
          for (int kpm = 0; kpm < kKernelPerMemory; ++kpm) {
            #pragma HLS UNROLL
          ReadASplit_KernelWidth:
            for (int kw = 0; kw < kKernelWidth; ++kw) {
              #pragma HLS UNROLL
              hlslib::WriteBlocking(aSplit[kpm * kKernelWidth + kw],
                                    static_cast<Data_t>(read[kpm][kw]),
                                    kTransposeDepth);
            }
          }
        }
      }
    }
  }
}

// Rotate between the different vertical buffers of A
void ReadARotate(hlslib::Stream<Data_t> aSplit[kMemoryWidth],
                 hlslib::Stream<Data_t> &aPipe) {
ReadARotate_Block_N:
  for (int bn = 0; bn < kBlocksN; ++bn) {
  ReadARotate_Block_P:
    for (int bp = 0; bp < kBlocksP; ++bp) {
    ReadARotate_M_Memory:
      for (int m = 0; m < kSizeMMemory; ++m) {
      ReadARotate_MemoryWidth:
        for (int mw = 0; mw < kMemoryWidth; ++mw) { 
        ReadARotate_N:
          for (int tn = 0; tn < kTileSizeN; ++tn) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE
            const auto read = hlslib::ReadBlocking(aSplit[mw]);
            hlslib::WriteBlocking(aPipe, read, 1);
          }
        }
      }
    }
  }
}
