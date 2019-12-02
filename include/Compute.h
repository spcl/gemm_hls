/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"

void ProcessingElement(Stream<ComputePackN_t, kPipeDepth> &aIn,
                       Stream<ComputePackN_t, kPipeDepth> &aOut,
                       Stream<ComputePackM_t, kPipeDepth> &bIn,
                       Stream<ComputePackM_t, kPipeDepth> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m);
