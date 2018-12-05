/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"
#include "hlslib/Stream.h"

constexpr unsigned kPipeDepth = 4;

static constexpr unsigned long kTotalReadsFromA =
    static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kOuterTileSizeN *
    kSizeK;
static constexpr unsigned long kTotalReadsFromB =
    static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kOuterTileSizeM *
    kSizeK;

using hlslib::Stream;

// Read wide bursts from memory, then distribute it into separate column
// buffers, which will be read out in column-major order and sent to the kernel
void ReadA(MemoryPack_t const a[],
           Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kMemoryWidth]);

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
#ifdef MM_CONVERT_A
void TransposeA(Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<Data_t, kPipeDepth> &toKernel);
#else
void TransposeA(Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<ComputePackN_t, kPipeDepth> &toKernel);
#endif

void ConvertWidthA(Stream<Data_t, kPipeDepth> &narrow,
                   Stream<ComputePackN_t, kPipeDepth> &wide);

void ReadB(MemoryPack_t const memory[],
           Stream<MemoryPack_t, 2 * kOuterTileSizeM> &pipe);

void ConvertWidthB(Stream<MemoryPack_t, 2 * kOuterTileSizeM> &wide,
                   Stream<ComputePackM_t> &narrow);

void ConvertWidthC(Stream<OutputPack_t> &narrow,
                   Stream<MemoryPack_t> &wide);

void WriteC(Stream<MemoryPack_t> &pipe, MemoryPack_t memory[]);

#ifndef MM_CONVERT_B
void FeedB(Stream<ComputePackM_t, 2 * kOuterTileSizeM> &fromMemory,
           Stream<ComputePackM_t, kPipeDepth> &toKernel);
#else
void FeedB(Stream<ComputePackM_t> &fromMemory,
           Stream<ComputePackM_t, kPipeDepth> &toKernel);
#endif
