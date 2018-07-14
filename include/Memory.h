/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "MatrixMatrix.h"
#include "hlslib/Stream.h"

static constexpr unsigned long kTotalReadsFromA =
    static_cast<unsigned long>(kOuterTilesN) * kOuterTilesM * kOuterTileSize *
    kSizeK;
static constexpr unsigned long kTotalReadsFromB = kTotalReadsFromA;

using hlslib::Stream;

// Read wide bursts from memory, then distribute it into separate column
// buffers, which will be read out in column-major order and sent to the kernel
void ReadA(MemoryPack_t const a[],
           Stream<Data_t, kOuterTileSize> aSplit[kMemoryWidth]);

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
void TransposeA(Stream<Data_t, kOuterTileSize> aSplit[kTransposeWidth],
                Stream<Data_t> &toKernel);

void ConvertWidthA(Stream<Data_t> &narrow, Stream<ComputePackN_t> &wide);

void ReadB(MemoryPack_t const memory[], Stream<MemoryPack_t> &pipe);

void ConvertWidthB(Stream<MemoryPack_t> &wide, Stream<ComputePackM_t> &narrow);

void FanInC(Stream<OutputPack_t> fromDrainers[kComputeTilesM],
            Stream<OutputPack_t> &toMemory);

void ConvertWidthC(Stream<OutputPack_t> &narrow, Stream<MemoryPack_t> &wide);

void WriteC(Stream<MemoryPack_t> &pipe, MemoryPack_t memory[]);
