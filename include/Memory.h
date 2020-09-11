/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"

#ifndef MM_TRANSPOSED_A

// Read wide bursts from memory, then distribute it into separate column
// buffers, which will be read out in column-major order and sent to the kernel
void ReadA(MemoryPackK_t const a[], Stream<Data_t> aSplit[kTransposeWidth],
           unsigned n, unsigned k, unsigned m);

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
#ifdef MM_CONVERT_A

void TransposeA(Stream<Data_t> aSplit[kTransposeWidth],
                Stream<Data_t> &toKernel, unsigned n, unsigned k, unsigned m);

void ConvertWidthA(Stream<Data_t> &narrow, Stream<ComputePackN_t> &wide,
                   unsigned, unsigned k, unsigned m);

#else

void TransposeA(Stream<Data_t> aSplit[kTransposeWidth],
                Stream<ComputePackN_t> &toKernel, unsigned n, unsigned k,
                unsigned m);
#endif

#else  // MM_TRANSPOSED_A

void ReadATransposed(MemoryPackN_t const memory[], Stream<MemoryPackN_t> &pipe,
                     const unsigned size_n, const unsigned size_k,
                     const unsigned size_m);

void ConvertWidthATransposed(Stream<MemoryPackN_t> &pipe_in,
                             Stream<ComputePackN_t> &pipe_out,
                             const unsigned size_n, const unsigned size_k,
                             const unsigned size_m);

#endif

void ReadB(MemoryPackM_t const memory[], Stream<MemoryPackM_t> &pipe,
           unsigned n, unsigned k, unsigned m);

#ifdef MM_CONVERT_B

void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
                   unsigned n, unsigned k, unsigned m);

void FeedB(Stream<ComputePackM_t> &converted, Stream<ComputePackM_t> &toKernel,
           unsigned n, unsigned k, unsigned m);

#else

void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           unsigned n, unsigned k, unsigned m);

#endif

void ConvertWidthC(Stream<OutputPack_t> &narrow, Stream<MemoryPackM_t> &wide,
                   unsigned n, unsigned k, unsigned m);

void WriteC(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[], unsigned n,
            unsigned k, unsigned m);
