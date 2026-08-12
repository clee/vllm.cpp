#include "scoped_server_thread.h"

#include <doctest/doctest.h>

#include <condition_variable>
#include <chrono>
#include <mutex>

using vllm::test::ScopedServerThread;

TEST_CASE("scoped server thread reports assertion failures without terminating") {
  std::mutex mutex;
  std::condition_variable changed;
  bool running = false;
  bool stopped = false;

  ScopedServerThread server_thread(
      [&]() {
        std::unique_lock<std::mutex> lock(mutex);
        running = true;
        changed.notify_all();
        changed.wait(lock, [&]() { return stopped; });
      },
      [&]() {
        std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
        changed.notify_all();
      });

  {
    std::unique_lock<std::mutex> lock(mutex);
    REQUIRE(changed.wait_for(lock, std::chrono::seconds(5),
                             [&]() { return running; }));
  }
  REQUIRE_MESSAGE(false, "scoped teardown fixture assertion");
}
