/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m);
