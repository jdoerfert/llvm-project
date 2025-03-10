#ifdef OBJSAN_TIME
#include <chrono>
#include <iostream>
#endif

class Timer {
public:
  Timer(const std::string &name = "Timer") {
#ifdef OBJSAN_TIME
    name_ = name;
    start_ = std::chrono::high_resolution_clock::now();
#endif
  }

  ~Timer() {
#ifdef OBJSAN_TIME
    auto end_ = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_)
            .count();
    std::cerr << name_ << ": " << duration << " microseconds" << std::endl;
#endif
  }

private:
#ifdef OBJSAN_TIME
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
#endif
};
