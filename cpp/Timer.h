#pragma once

#include <chrono>

namespace cpputils {

class Timer {

public:
  inline Timer();

  inline void Start();

  inline double Stop();

  inline double Elapsed() const;

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  double elapsed_{};
};

Timer::Timer() : start_(std::chrono::high_resolution_clock::now()) {}

void Timer::Start() { start_ = std::chrono::high_resolution_clock::now(); }

double Timer::Stop() {
  const auto end = std::chrono::high_resolution_clock::now();
  elapsed_ = 1e-9 *
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
                 .count();
  return elapsed_;
}

double Timer::Elapsed() const { return elapsed_; }

} // End namespace cpputils
