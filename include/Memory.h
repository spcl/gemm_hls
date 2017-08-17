/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      August 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "MatrixMatrix.h"
#include "hlslib/Stream.h"

void ReadASplit(MemoryPack_t const a[],
                hlslib::Stream<Data_t> aSplit[kMemoryWidth]);

void ReadARotate(hlslib::Stream<Data_t> aSplit[kMemoryWidth],
                 hlslib::Stream<Data_t> &aPipe);

void ReadBMemory(MemoryPack_t const b[], hlslib::Stream<MemoryPack_t> &bPipe);

void ReadBKernel(hlslib::Stream<MemoryPack_t> &bMem,
                 hlslib::Stream<KernelPack_t> &bPipe);

void WriteCKernel(hlslib::Stream<KernelPack_t> &cPipe,
                  hlslib::Stream<MemoryPack_t> &cMem);

void WriteCMemory(hlslib::Stream<MemoryPack_t> &cMem, MemoryPack_t c[]);
