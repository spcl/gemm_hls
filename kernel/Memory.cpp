/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "Memory.h"
#include <cassert>

using hlslib::Stream;

#ifndef MM_TRANSPOSED_A

unsigned IndexA(const unsigned n0, const unsigned n1, const unsigned n2,
                const unsigned k0, const unsigned k1, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
  #pragma HLS INLINE
  const auto index =
      (n0 * kOuterTileSizeN + n1 * kInnerTileSizeN + n2) * SizeKMemory(size_k) +
      (k0 * (kTransposeWidth / kMemoryWidthK) + k1);
  // assert(index < size_n * SizeKMemory(size_k));
  return index;
}

#else  // MM_TRANSPOSED_A

unsigned IndexATransposed(const unsigned k, const unsigned n0,
                          const unsigned n1m, const unsigned size_n,
                          const unsigned size_k, const unsigned size_m) {
  #pragma HLS INLINE
  const auto index =
      k * SizeNMemory(size_n) + (n0 * kOuterTileSizeNMemory + n1m);
  // assert(index < size_k * SizeNMemory(size_n));
  return index;
}

#endif  // MM_TRANSPOSED_A

unsigned IndexB(const unsigned k, const unsigned m0, const unsigned m1m,
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {
  #pragma HLS INLINE
  const auto index =
      k * SizeMMemory(size_m) + (m0 * kOuterTileSizeMMemory + m1m);
  // assert(index < size_k * SizeMMemory(size_m));
  return index;
}

unsigned IndexC(const unsigned n0, const unsigned n1, const unsigned m0,
                const unsigned m1m, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
  #pragma HLS INLINE
  const auto index = (n0 * kOuterTileSizeN + n1) * SizeMMemory(size_m) +
                     (m0 * kOuterTileSizeMMemory + m1m);
  // assert(index < size_n * SizeMMemory(size_m));
  return index;
}

#ifndef MM_TRANSPOSED_A

void _ReadAInner(MemoryPackK_t const a[],
                 Stream<Data_t> aSplit[kTransposeWidth], const unsigned n0,
                 const unsigned n1, const unsigned n2, const unsigned k0,
                 const unsigned k1, const unsigned size_n,
                 const unsigned size_k, const unsigned size_m) {
  #pragma HLS INLINE
  auto pack = a[IndexA(n0, n1, n2, k0, k1, size_n, size_k, size_m)];
ReadA_Unroll:
  for (unsigned w = 0; w < kMemoryWidthK; ++w) {
    #pragma HLS UNROLL
    aSplit[k1 * kMemoryWidthK + w].Push(pack[w]);
  }
}

template <unsigned innerReads>
void _ReadAInnerLoop(MemoryPackK_t const a[],
                     Stream<Data_t> aSplit[kTransposeWidth], unsigned n0,
                     unsigned n1, unsigned k0, const unsigned size_n,
                     const unsigned size_k, const unsigned size_m) {
  #pragma HLS INLINE
ReadA_N2:
  for (unsigned n2 = 0; n2 < kInnerTileSizeN; ++n2) {
  ReadA_TransposeWidth:
    for (unsigned k1 = 0; k1 < (kTransposeWidth / kMemoryWidthK); ++k1) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      _ReadAInner(a, aSplit, n0, n1, n2, k0, k1, size_n, size_k, size_m);
    }
  }
}

// Need a special case for kMemoryWidthK == kTransposeWidth, as Vivado HLS
// otherwise doesn't pipeline the loops (because the inner trip count is 1).
template <>
void _ReadAInnerLoop<1>(MemoryPackK_t const a[],
                        Stream<Data_t> aSplit[kTransposeWidth],
                        const unsigned n0, const unsigned n1, const unsigned k0,
                        const unsigned size_n, const unsigned size_k,
                        const unsigned size_m) {
  #pragma HLS INLINE
ReadA_N2:
  for (unsigned n2 = 0; n2 < kInnerTileSizeN; ++n2) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    _ReadAInner(a, aSplit, n0, n1, n2, k0, 0, size_n, size_k, size_m);
  }
}

void ReadA(MemoryPackK_t const a[], Stream<Data_t> aSplit[kTransposeWidth],
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m) {
  assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
          OuterTilesM(size_m) * (size_k / kTransposeWidth) * kInnerTilesN *
          kInnerTileSizeN * (kTransposeWidth / kMemoryWidthK) *
          MemoryPackK_t::kWidth) == TotalReadsFromA(size_n, size_k, size_m));

ReadA_N0:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  ReadA_M0:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
    ReadA_K0:
      for (unsigned k0 = 0; k0 < size_k / kTransposeWidth; ++k0) {
      ReadA_N1:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
          _ReadAInnerLoop<kTransposeWidth / kMemoryWidthK>(
              a, aSplit, n0, n1, k0, size_n, size_k, size_m);
        }
      }
    }
  }
}

template <unsigned inner_tiles>
void _TransposeAInner(Stream<Data_t> aSplit[kTransposeWidth],
                      Stream<ComputePackN_t> &toKernel, const unsigned k) {
  #pragma HLS INLINE
  for (unsigned n1 = 0; n1 < kOuterTileSizeN / kComputeTileSizeN; ++n1) {
    ComputePackN_t pack;
  TransposeA_N2:
    for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      pack[n2] = aSplit[k % kTransposeWidth].Pop();
      // Pop from each stream kOuterTileSizeN times in a row
      if (n2 == kComputeTileSizeN - 1) {
        toKernel.Push(pack);
      }
    }
  }
}

template <>
void _TransposeAInner<1>(Stream<Data_t> aSplit[kTransposeWidth],
                         Stream<ComputePackN_t> &toKernel, const unsigned k) {
  #pragma HLS INLINE
  for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    ComputePackN_t pack;
    pack[0] = aSplit[k % kTransposeWidth].Pop();
    toKernel.Push(pack);
  }
}

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
void TransposeA(Stream<Data_t> aSplit[kTransposeWidth],
                Stream<ComputePackN_t> &toKernel, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
  assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
          OuterTilesM(size_m) * size_k * kOuterTileSizeN) ==
         TotalReadsFromA(size_n, size_k, size_m));

TransposeA_N0:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  TransposeA_M0:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
    TransposeA_K:
      for (unsigned k = 0; k < size_k; ++k) {
        _TransposeAInner<kComputeTileSizeN>(aSplit, toKernel, k);
      }
    }
  }
}

#ifdef MM_CONVERT_A
void ConvertWidthA(Stream<Data_t> &narrow, Stream<ComputePackN_t> &wide,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m) {
ConvertWidthA_Outer:
  for (unsigned i = 0;
       i < TotalReadsFromA(size_n, size_k, size_m) / ComputePackN_t::kWidth;
       ++i) {
    ComputePackN_t pack;
  ConvertWidthA_Compute:
    for (unsigned w = 0; w < ComputePackN_t::kWidth; ++w) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      pack[w] = narrow.Pop();
    }
    wide.Push(pack);
  }
}
#endif

#else  // MM_TRANSPOSED_A == true

void ReadATransposed(MemoryPackN_t const memory[], Stream<MemoryPackN_t> &pipe,
                     const unsigned size_n, const unsigned size_k,
                     const unsigned size_m) {
  assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
          OuterTilesM(size_m) * size_k * kOuterTileSizeNMemory *
          MemoryPackN_t::kWidth) == TotalReadsFromA(size_n, size_k, size_m));

ReadA_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  ReadA_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
    ReadA_K:
      for (unsigned k = 0; k < size_k; ++k) {
      ReadA_BufferA_N1:
        for (unsigned n1m = 0; n1m < kOuterTileSizeNMemory; ++n1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          pipe.Push(
              memory[IndexATransposed(k, n0, n1m, size_n, size_k, size_m)]);
        }
      }
    }
  }
}

void ConvertWidthATransposed(Stream<MemoryPackN_t> &wide,
                             Stream<ComputePackN_t> &narrow,
                             const unsigned size_n, const unsigned size_k,
                             const unsigned size_m) {
  static_assert(kMemoryWidthN % kComputeTileSizeN == 0,
                "Tile size must be a multiple of memory width.");

#ifdef MM_CONVERT_A
ConvertWidthA_Outer:
  for (unsigned i = 0;
       i < TotalReadsFromA(size_n, size_k, size_m) / kMemoryWidthN; ++i) {
    MemoryPackN_t memoryPack;
  ConvertWidthA_Memory:
    for (unsigned j = 0; j < kMemoryWidthN / kComputeTileSizeN; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      if (j == 0) {
        memoryPack = wide.Pop();
      }
      ComputePackN_t computePack;
    ConvertWidthA_Compute:
      for (unsigned w = 0; w < kComputeTileSizeN; ++w) {
        #pragma HLS UNROLL
        computePack[w] = memoryPack[j * kComputeTileSizeN + w];
      }
      narrow.Push(computePack);
    }
#else
  narrow.Push(wide.Pop());
#endif
  }
}

#endif  // MM_TRANSPOSED_A == true

void ReadB(MemoryPackM_t const memory[], Stream<MemoryPackM_t> &pipe,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m) {
  assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
          OuterTilesM(size_m) * size_k * kOuterTileSizeMMemory *
          MemoryPackM_t::kWidth) == TotalReadsFromB(size_n, size_k, size_m));

ReadB_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  ReadB_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
    ReadB_K:
      for (unsigned k = 0; k < size_k; ++k) {
      ReadB_BufferB_M1:
        for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          pipe.Push(memory[IndexB(k, m0, m1m, size_n, size_k, size_m)]);
        }
      }
    }
  }
}

void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m) {
  assert(kMemoryWidthM % kComputeTileSizeM == 0);

  assert(((TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM) *
          MemoryPackM_t::kWidth) == TotalReadsFromB(size_n, size_k, size_m));

  assert(((TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM) *
          (kMemoryWidthM / kComputeTileSizeM) * ComputePackM_t::kWidth) ==
         TotalReadsFromB(size_n, size_k, size_m));

ConvertWidthB_Outer:
  for (unsigned i = 0;
       i < TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM; ++i) {
    MemoryPackM_t memoryPack;
  ConvertWidthB_Memory:
    for (unsigned j = 0; j < kMemoryWidthM / kComputeTileSizeM; ++j) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      if (j == 0) {
        memoryPack = wide.Pop();
      }
      ComputePackM_t computePack;
    ConvertWidthB_Compute:
      for (unsigned w = 0; w < kComputeTileSizeM; ++w) {
        #pragma HLS UNROLL
        computePack[w] = memoryPack[j * kComputeTileSizeM + w];
      }
      narrow.Push(computePack);
    }
  }
}

void ConvertWidthC(Stream<ComputePackM_t> &narrow, Stream<MemoryPackM_t> &wide,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m) {
  assert(kMemoryWidthM % ComputePackM_t::kWidth == 0);

  // assert((((size_n * size_m) / MemoryPackM_t::kWidth) *
  //         (kMemoryWidthM / ComputePackM_t::kWidth) * ComputePackM_t::kWidth) ==
  //        size_n * size_m);

ConvertWidthC_N:
  for (unsigned i = 0; i < OuterTilesN(size_n) * kOuterTileSizeN; ++i) {
  ConvertWidthC_M:
    for (unsigned j = 0; j < OuterTilesM(size_m) * kOuterTileSizeMMemory; ++j) {
#ifdef MM_CONVERT_B
    ConvertWidthB_Memory:
      MemoryPackM_t memoryPack;
      for (unsigned j = 0; j < kMemoryWidthM / ComputePackM_t::kWidth; ++j) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN
        const auto computePack = narrow.Pop();
      ConvertWidthB_Compute:
        for (unsigned w = 0; w < ComputePackM_t::kWidth; ++w) {
          #pragma HLS UNROLL
          memoryPack[j * ComputePackM_t::kWidth + w] = computePack[w];
        }
        if (j == kMemoryWidthM / ComputePackM_t::kWidth - 1) {
          wide.Push(memoryPack);
        }
      }
#else
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_FLATTEN
      wide.Push(narrow.Pop());
#endif
    }
  }
}

void WriteC(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[],
            const unsigned size_n, const unsigned size_k,
            const unsigned size_m) {
  // assert((OuterTilesN(size_n) * OuterTilesM(size_m) * kOuterTileSizeN *
  //         kOuterTileSizeMMemory * MemoryPackM_t::kWidth) == size_n * size_m);

WriteC_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  WriteC_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
    WriteC_N1:
      for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1) {
      WriteC_M1:
        for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN
          const auto val = pipe.Pop();
          if ((n0 * kOuterTileSizeN + n1 < size_n) &&
              (m0 * kOuterTileSizeMMemory + m1m < SizeMMemory(size_m))) {
            memory[IndexC(n0, n1, m0, m1m, size_n, size_k, size_m)] = val;
          }
        }
      }
#ifndef MM_SYNTHESIS
      std::cout << "Finished tile (" << n0 << ", " << m0 << ") of ("
                << OuterTilesN(size_n) - 1 << ", " << OuterTilesM(size_m) - 1
                << ")\n"
                << std::flush;
#endif
    }
  }
}

#ifndef MM_CONVERT_B
void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m) {
#else
void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m) {
#endif

  assert(static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
             size_k * kInnerTilesM * ComputePackM_t::kWidth ==
         TotalReadsFromB(size_n, size_k, size_m));

  const unsigned bound_n = OuterTilesN(size_n);
  const unsigned bound_m = OuterTilesM(size_m);

FeedB_OuterTile_N:
  for (unsigned n0 = 0; n0 < bound_n; ++n0) {
  FeedB_OuterTile_M:
    for (unsigned m0 = 0; m0 < bound_m; ++m0) {
    FeedB_K:
      for (unsigned k = 0; k < size_k; ++k) {
        ComputePackM_t buffer[kInnerTilesM];

      FeedB_Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
        FeedB_Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            ComputePackM_t val;
            if (n1 == 0) {
              val = fromMemory.Pop();
              buffer[m1] = val;
            } else {
              val = buffer[m1];
            }
            toKernel.Push(val);
          }
        }
      }
    }
  }
}
