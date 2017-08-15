/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      April 2016
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <cstddef> // ap_int.h will break some compilers if this is not included 
#include <ap_int.h>

namespace hlslib {

namespace {

template <typename T, int width>
class DataPackProxy; // Forward declaration

} // End anonymous namespace

/// Class to accommodate SIMD-style vectorization of a data path on FPGA using
/// ap_uint to force wide ports.
///
/// DataPacks can be nested, e.g. DataPack<DataPack<int, 4>, 4>, but the proxy
/// class can cause some issues when using nested indices to assign values.
template <typename T, int width>
class DataPack {

  static_assert(width > 0, "Width must be positive");

public:

  static constexpr int kBits = 8 * sizeof(T);
  using Pack_t = ap_uint<kBits>;
  using Internal_t = ap_uint<width * kBits>;

  DataPack() : data_() {}

  DataPack(DataPack<T, width> const &other) = default;

  DataPack(DataPack<T, width> &&other) = default;

  DataPack(T const &value) : data_() {
    #pragma HLS INLINE
    Fill(value);
  }

  DataPack(T const arr[width]) : data_() { 
    #pragma HLS INLINE
    Pack(arr);
  }

  DataPack<T, width>& operator=(DataPack<T, width> &&other) {
    #pragma HLS INLINE
    data_ = other.data_;
    return *this;
  }

  DataPack<T, width>& operator=(DataPack<T, width> const &other) {
    #pragma HLS INLINE
    data_ = other.data_;
    return *this;
  }

  T Get(int i) const {
    #pragma HLS INLINE
    Pack_t temp = data_.range((i + 1) * kBits - 1, i * kBits);
    return *reinterpret_cast<T const *>(&temp);
  }

  void Set(int i, T value) {
    #pragma HLS INLINE
    Pack_t temp = *reinterpret_cast<Pack_t const *>(&value);
    data_.range((i + 1) * kBits - 1, i * kBits) = temp;
  }

  void Fill(T const &value) {
    #pragma HLS INLINE
  DataPack_Fill:
    for (int i = 0; i < width; ++i) {
      #pragma HLS UNROLL
      Set(i, value);
    }
  }

  void Pack(T const arr[width]) {
    #pragma HLS INLINE
  DataPack_Pack:
    for (int i = 0; i < width; ++i) {
      #pragma HLS UNROLL
      Set(i, arr[i]);
    }
  }

  void Unpack(T arr[width]) const {
    #pragma HLS INLINE
  DataPack_Unpack:
    for (int i = 0; i < width; ++i) {
      #pragma HLS UNROLL
      arr[i] = Get(i);
    }
  }

  void operator<<(T const arr[width]) {
    #pragma HLS INLINE
    Pack(arr);
  }

  void operator>>(T arr[width]) const {
    #pragma HLS INLINE
    Unpack(arr);
  }

  T operator[](const size_t i) const {
    #pragma HLS INLINE
    return Get(i);    
  }

  DataPackProxy<T, width> operator[](const size_t i);

  // Access to internal data directly if necessary
  Internal_t &data() { return data_; }
  Internal_t data() const { return data_; }

  /// Copy values from this DataPack into another DataPack, starting from
  /// position src into position dst, and copying count elements.
  template <unsigned src, unsigned dst, unsigned count, int otherWidth>
  void ShiftTo(DataPack<T, otherWidth> &other) const {
    #pragma HLS INLINE
    static_assert(src + count <= width && dst + count <= otherWidth,
                  "Invalid range");
  DataPack_Shift:
    for (int i = 0, s = src, d = dst; i < count; ++i, ++s, ++d) {
      #pragma HLS UNROLL
      other.Set(d, Get(s));
    }
  }

private:

  Internal_t data_;

};

namespace {

/// Proxy class to allow assigning values to individual elements of the DataPack
/// directly.
template <typename T, int width>
class DataPackProxy {

  static constexpr int kBits = 8 * width;

public:

  DataPackProxy(DataPack<T, width> &data, int index)
      : index_(index), data_(data) {
    #pragma HLS INLINE
  }

  DataPackProxy(DataPackProxy<T, width> const &other) = default; 

  DataPackProxy(DataPackProxy<T, width> &&) = default;

  ~DataPackProxy() {}

  void operator=(T const &rhs) {
    #pragma HLS INLINE
    data_.Set(index_, rhs);
  }

  void operator=(T &&rhs) {
    #pragma HLS INLINE
    data_.Set(index_, rhs);
  }

  void operator=(DataPackProxy<T, width> const &rhs) {
    #pragma HLS INLINE
    data_.Set(index_, static_cast<T>(rhs));
  }

  void operator=(DataPackProxy<T, width> &&rhs) {
    #pragma HLS INLINE
    data_.Set(index_, static_cast<T>(rhs));
  }

  operator T() const {
    #pragma HLS INLINE
    return data_.Get(index_);
  }

private:

  int index_;
  DataPack<T, width> &data_;

};

} // End anonymous namespace

template <typename T, int width>
DataPackProxy<T, width> DataPack<T, width>::operator[](const size_t i) {
  #pragma HLS INLINE
  return DataPackProxy<T, width>(*this, i);
}

template <typename T, int width>
std::ostream& operator<<(std::ostream &os, DataPack<T, width> const &rhs) {
  os << "{" << rhs.Get(0);
  for (int i = 1; i < width; ++i) {
    os << ", " << rhs.Get(i);
  }
  os << "}";
  return os;
}

} // End namespace hlslib 
