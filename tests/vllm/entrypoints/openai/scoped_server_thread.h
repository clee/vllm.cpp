#pragma once

#include <functional>
#include <thread>
#include <utility>

namespace vllm::test {

// Owns a test server's background thread. Fatal doctest assertions unwind the
// test body, so teardown must stop the server and join before std::thread's
// destructor observes a joinable thread.
class ScopedServerThread {
 public:
  template <typename Start, typename Stop>
  ScopedServerThread(Start&& start, Stop&& stop)
      : stop_(std::forward<Stop>(stop)),
        thread_(std::forward<Start>(start)) {}

  ~ScopedServerThread() {
    stop_();
    if (thread_.joinable()) thread_.join();
  }

  ScopedServerThread(const ScopedServerThread&) = delete;
  ScopedServerThread& operator=(const ScopedServerThread&) = delete;

 private:
  std::function<void()> stop_;
  std::thread thread_;
};

}  // namespace vllm::test
