#include <chrono>
#include <iostream>

class Timer {
public:
  Timer(const std::string &name = "Timer")
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {
    auto end_ = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_)
            .count();
    std::cerr << name_ << ": " << duration << " microseconds" << std::endl;
  }

private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

