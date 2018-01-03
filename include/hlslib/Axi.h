/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      October 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <ap_int.h>

namespace hlslib {

namespace axi {

/// Implements the AXI Stream interface, which is inferred by the name of the
/// variables (data, keep and last).
template <typename T>
struct Stream {

  T data;
  ap_uint<sizeof(T)> keep{~(keep & 0)}; // Set all bits to 1
  ap_uint<1> last;

  Stream() : data(), keep(0), last(1) {}
  Stream(decltype(data) const &_data, decltype(last) const &_last)
      : data(_data), last(_last) {}
  Stream(decltype(data) const &_data) : data(_data), last(1) {}
};

/// Implements the command bus interface for the DataMover IP
/// https://www.xilinx.com/support/documentation/ip_documentation/axi_datamover/v5_1/pg022_axi_datamover.pdf
template <size_t addressWidth, size_t bttWidth>
struct Command {

  // Typically addressWith is in {32, 64} and bttWidth is 23, so the whole bus
  // is 72 or 104 bits.
  ap_uint<bttWidth> length; // BTT (bytes to transfer)
  ap_uint<1> type{1};
  ap_uint<6> dsa{0};
  ap_uint<1> eof{1};
  ap_uint<1> drr{1};
  ap_uint<addressWidth> address;
  ap_uint<4> tag{0};
  ap_uint<4> reserved{0};

  Command(decltype(address) const &_address, decltype(length) const &_length)
      : length(_length), address(_address) {}
  Command() : length(0), address(0) {}
};

/// Implements the status bus interface for the DataMover IP
/// https://www.xilinx.com/support/documentation/ip_documentation/axi_datamover/v5_1/pg022_axi_datamover.pdf
struct Status { // 8 bits

  ap_uint<4> tag{0};
  ap_uint<1> internalError{0};
  ap_uint<1> decodeError{0};
  ap_uint<1> slaveError{0};
  ap_uint<1> okay;

	Status(bool _okay) : okay(_okay) {}  
  Status() : okay(true) {}
};

} // End namespace axi 

} // End namespace hlslib
