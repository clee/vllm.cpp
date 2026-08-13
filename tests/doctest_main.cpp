#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {

[[noreturn]] void DiagnosticTerminate() noexcept {
  const std::exception_ptr exception = std::current_exception();
  if (exception == nullptr) {
    std::fputs(
        "[vllm-test-probe] std::terminate current_exception=none\n", stderr);
  } else {
    std::fputs(
        "[vllm-test-probe] std::terminate current_exception=present\n",
        stderr);
    try {
      std::rethrow_exception(exception);
    } catch (const std::exception& error) {
      std::fprintf(stderr,
                   "[vllm-test-probe] std::terminate std::exception what=%s\n",
                   error.what());
    } catch (...) {
      std::fputs(
          "[vllm-test-probe] std::terminate exception=non-std-exception\n",
          stderr);
    }
  }
  std::fflush(stderr);
  std::abort();
}

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate(DiagnosticTerminate);
  return doctest::Context(argc, argv).run();
}
