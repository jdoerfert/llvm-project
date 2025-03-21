#include <chrono>
#include <iostream>
#include <mutex>

class Timer {
public:
  Timer(const std::string &name = "InputGenTimer")
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);

    auto end_ = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_)
            .count();
    std::cerr << "InputGenTimer " << name_ << ": " << duration << " nanoseconds"
              << std::endl;
  }

private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

