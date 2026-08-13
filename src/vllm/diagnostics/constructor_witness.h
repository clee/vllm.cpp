#ifndef VLLM_DIAGNOSTICS_CONSTRUCTOR_WITNESS_H_
#define VLLM_DIAGNOSTICS_CONSTRUCTOR_WITNESS_H_

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace vllm::diagnostics {

constexpr const char* kConstructorWitnessEnvironment = "VLLM_WINDOWS_CTOR_DIAGNOSTIC";

inline bool ConstructorWitnessEnabled() noexcept {
  const char* value = std::getenv(kConstructorWitnessEnvironment);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline void ConstructorWitness(const char* function, const char* stage,
                               const char* phase,
                               long long index = -1) noexcept {
  if (!ConstructorWitnessEnabled()) return;
  if (index < 0) {
    std::fprintf(stderr,
                 "VLLM_CTOR_DIAGNOSTIC: function=%s stage=%s phase=%s\n",
                 function, stage, phase);
  } else {
    std::fprintf(
        stderr,
        "VLLM_CTOR_DIAGNOSTIC: function=%s stage=%s phase=%s index=%lld\n",
        function, stage, phase, index);
  }
  std::fflush(stderr);
}

inline void ConstructorWitnessBefore(const char* function, const char* stage,
                                     long long index = -1) noexcept {
  ConstructorWitness(function, stage, "before", index);
}

inline void ConstructorWitnessAfter(const char* function, const char* stage,
                                    long long index = -1) noexcept {
  ConstructorWitness(function, stage, "after", index);
}

// A mem-initializer is one full expression. This temporary is constructed before
// that expression and destroyed only after the target member was successfully
// initialized. During exception unwinding it deliberately omits "after", so the
// first unmatched marker remains the failing initializer.
class ConstructorWitnessPhase {
 public:
  ConstructorWitnessPhase(const char* function, const char* stage) noexcept
      : function_(function),
        stage_(stage),
        enabled_(ConstructorWitnessEnabled()),
        uncaught_exceptions_(std::uncaught_exceptions()) {
    if (enabled_) ConstructorWitnessBefore(function_, stage_);
  }

  ConstructorWitnessPhase(const ConstructorWitnessPhase&) = delete;
  ConstructorWitnessPhase& operator=(const ConstructorWitnessPhase&) = delete;

  ~ConstructorWitnessPhase() noexcept {
    if (enabled_ && std::uncaught_exceptions() == uncaught_exceptions_) {
      ConstructorWitnessAfter(function_, stage_);
    }
  }

 private:
  const char* function_;
  const char* stage_;
  bool enabled_;
  int uncaught_exceptions_;
};

}  // namespace vllm::diagnostics

#endif  // VLLM_DIAGNOSTICS_CONSTRUCTOR_WITNESS_H_
