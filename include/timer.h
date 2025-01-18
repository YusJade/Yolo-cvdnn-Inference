#pragma once

#include <chrono>

class Timer {
 public:
  inline void Tik() { start = std::chrono::high_resolution_clock::now(); }

  inline void Tok() { end = std::chrono::high_resolution_clock::now(); }

  inline float FPS() {
    duration = end - start;
    return 1 / duration.count();
  }

  inline double CostTime() { return duration.count(); }

 private:
  std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> end;
  std::chrono::duration<double> duration;
};