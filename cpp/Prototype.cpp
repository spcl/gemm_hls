#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "Timer.h"

constexpr int kSize = 256;
constexpr int kTileSizeN = 64;
constexpr int kTileSizeP = 64;
constexpr int kBlocksN = kSize / kTileSizeN;
constexpr int kBlocksP = kSize / kTileSizeP;

inline int GlobalIndex(int bn, int bp, int tn, int tp) {
  return (bn * kTileSizeN + tn) * kSize + bp * kTileSizeP + tp;
}

void Tiled(float const a[], float const b[], float c[]) {

  for (int bn = 0; bn < kBlocksN; ++bn) {
    for (int bp = 0; bp < kBlocksP; ++bp) {

      float cLocal[kTileSizeN][kTileSizeP] = {0}; // Local result buffer

      // Loop over collapsed dimension
      for (int m = 0; m < kSize; ++m) {

        // Load column of A
        float aColumn[kTileSizeN]; // Local buffer of A
        for (int tn = 0; tn < kTileSizeN; ++tn) {
          #pragma HLS PIPELINE
          aColumn[tn] = a[GlobalIndex(bn, 0, tn, m)]; 
        }

        // Stream row of B and apply it to all correspondings elements of A 
        for (int tp = 0; tp < kTileSizeP; ++tp) {
          #pragma HLS PIPELINE
          const auto readB = b[GlobalIndex(0, bp, m, tp)];
          for (int tn = 0; tn < kTileSizeN; ++tn) {
            #pragma HLS UNROLL
            cLocal[tn][tp] += readB * aColumn[tn];
          }
        }

      }

      // Write out result block
      for (int tn = 0; tn < kTileSizeN; ++tn) {
        for (int tp = 0; tp < kTileSizeP; ++tp) {
          #pragma HLS PIPELINE
          c[GlobalIndex(bn, bp, tn, tp)] = cLocal[tn][tp];
        }
      }
    }
  }
}

void Reference(float const a[], float const b[], float c[]) {
  for (int n = 0; n < kSize; ++n) {
    for (int p = 0; p < kSize; ++p) {
      c[n * kSize + p] = 0;
      for (int m = 0; m < kSize; ++m) {
        c[n * kSize + p] += a[n * kSize + m] * b[m * kSize + p];
      }
    }
  }
}

int main() {
  std::vector<float> a(kSize*kSize);
  std::vector<float> b(kSize*kSize);
  std::vector<float> cReference(kSize*kSize, 0);
  std::vector<float> cTiled(kSize*kSize, 0);
  std::random_device rd;
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist;
  std::for_each(a.begin(), a.end(), [&](float &i) { i = dist(rng); });
  std::for_each(b.begin(), b.end(), [&](float &i) { i = dist(rng); });
  cpputils::Timer t;
  Reference(a.data(), b.data(), cReference.data());
  const auto tReference = t.Stop();
  t.Start();
  Tiled(a.data(), b.data(), cTiled.data());
  const auto tTiled = t.Stop();
  for (int i = 0; i < kSize * kSize; ++i) {
    const auto diff = std::abs(cReference[i] - cTiled[i]);
    if (diff >= 1e-3) {
      std::cout << "Mismatch at (" << i / kSize << ", " << i % kSize
                << "): " << cTiled[i] << " (should be " << cReference[i]
                << ").\n";
    }
  }
  std::cout << "Reference: " << tReference << " seconds.\n";
  std::cout << "Tiled:     " << tTiled << " seconds.\n";
  return 0;
}
