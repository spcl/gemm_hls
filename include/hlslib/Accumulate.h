/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "hlslib/DataPack.h"
#include "hlslib/Operators.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"

// This header includes a module that allows pipelined accumulation of data
// types with a non-zero latency on the operation, such as addition of floating
// point numbers, using a minimal amount of arithmetic units.
// The functionality is split into three functions:
//
//   1) Iterate, which accumulates `latency` different inputs, outputting
//      partial results.
//   2) Feedback, which loops partial results back to Iterate to continue
//      accumulation until done, then emits the result.
//   3) Reduce, which finally collapses the `latency` inputs into a single
//      output. 
//
// If latency is not set high enough, the design will hang, as no new values
// will be available from Feedback when Iterate requires them, and the pipeline
// will stall forever. The latency should be at least the sum of the latencies
// of the Iterate and Feedback modules, plus a few cycles for the overhead of
// the FIFOs between them.
// For example, for single precision floating point addition at 300 MHz, the
// latency of Iterate is 10 and the latency of bounce is 2. A latency of 14 is 
// enough to successfully run accumulation.
//
// Due to a bug where Vivado HLS (as of 2017.1) considers all streams static,
// the three functions cannot be factored into a generic function here, as
// calling it multiple times would result in dataflow errors on the internal
// streams. Instead call them as follows:
//
//   Stream<T> toFeedback("fromFeedback");
//   Stream<T> fromFeedback("fromFeedback");
//   #pragma HLS STREAM variable=fromFeedback depth=latency
//   Stream<T> toReduce("toReduce");
//   HLSLIB_DATAFLOW_FUNCTION(AccumulateIterate, in, fromFeedback, toFeedback);
//   HLSLIB_DATAFLOW_FUNCTION(AccumulateFeedback, toFeedback, fromFeedback,
//                            toReduce);
//   AccumulateReduce(toReduce, out);
//
// For convenience, a function for reducing with a single cycle latency
// operation is also included as AccumulateSimple.

namespace hlslib {

template <typename T, class Operator, int latency, int size, int iterations>
void AccumulateIterate(Stream<T> &input, Stream<T> &fromFeedback,
                       Stream<T> &toFeedback) {
AccumulateIterate_Iterations:
  for (int i = 0; i < iterations; ++i) {
  AcumulateIterate_Size:
    for (int j = 0; j < size / latency; ++j) {
    AccumulateIterate_Latency:
      for (int k = 0; k < latency; ++k) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_FLATTEN
        const T a = hlslib::ReadBlocking(input);
        T b;
        if (j > 0) {
          b = hlslib::ReadOptimistic(fromFeedback);
        } else {
          b = Operator::identity();
        }
        const auto result = Operator::Apply(a, b);
        hlslib::WriteBlocking(toFeedback, result, 1);
      }
    }
  }
}

template <typename T, class Operator, int latency, int size, int iterations>
void AccumulateFeedback(Stream<T> &toFeedback, Stream<T> &fromFeedback,
                        Stream<T> &toReduce) {
AccumulateFeedback_Iterations:
  for (int i = 0; i < iterations; ++i) {
  AccumulateFeedback_Size:
    for (int j = 0; j < size / latency; ++j) {
    AccumulateFeedback_Latency:
      for (int k = 0; k < latency; ++k) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_FLATTEN
        const auto read = hlslib::ReadBlocking(toFeedback);
        if (j < size / latency - 1) {
          // Feedback back
          hlslib::WriteBlocking(fromFeedback, read, latency);
        } else {
          // Write to output
          hlslib::WriteBlocking(toReduce, read, 1);
        }
      }
    }
  }
}

template <typename T, class Operator, int latency, int size, int iterations>
void AccumulateReduce(Stream<T> &toReduce, Stream<T> &output) {
AccumulateReduce_Iterations:
  for (int i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=latency
    T result(Operator::identity());
  AccumulateReduce_Latency:
    for (int j = 0; j < latency; ++j) {
      const auto read = hlslib::ReadBlocking(toReduce);
      result = Operator::Apply(result, read);
    }
    hlslib::WriteBlocking(output, result, 1);
  }
}

/// Trivial implementation for single cycle latency
template <typename T, class Operator, int size, int iterations>
void AccumulateSimple(Stream<T> &in, Stream<T> &out) {
AccumulateSimple_Iterations:
  for (int i = 0; i < iterations; ++i) {
    T acc;
  AccumulateSimple_Size:
    for (int j = 0; j < size; ++j) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE
      const auto a = hlslib::ReadBlocking(in);
      const auto b = (j > 0) ? acc : T(Operator::identity());
      acc = Operator::Apply(a, b);
      if (j == size - 1) {
        hlslib::WriteBlocking(out, acc, 1);
      }
    }
  }
}

} // End namespace hlslib
