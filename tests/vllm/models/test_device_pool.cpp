// vllm.cpp original (the shared scratch pool is a vt-runtime deviation, porting
// inventory §9.1); vLLM has no mirror because it never had this design — its
// allocation handle carries the device as field 0
// (`vllm/device_allocator/__init__.py:12-14` @ pin 555967922) and torch's cache
// is per-device by construction (`c10/cuda/CUDACachingAllocator.h:118-172`).
//
// THE GATE FOR #516 (.agents/specs/pool-device-key.md): `vllm::Pool()` was a
// process-wide free list keyed by BYTE SIZE CLASS ONLY. The device was not in
// the key, so a block allocated through one backend was handed to a `DBuf`
// running on another. One fault, two symptoms, chosen by direction:
//
//   cudaMalloc block -> CPU-backend forward : SIGSEGV, compute-sanitizer CLEAN
//                                             (the fault is host-side)
//   host block       -> CUDA forward        : SILENT all-NaN output
//
// This file is the DIRECT gate. It needs no GPU, no checkpoint and no NAS: two
// distinguishable fake backends stand in for two devices (the technique
// tests/vt/test_backend_multidevice.cpp and tests/vt/test_reference_tier.cpp
// already use), and every case goes through `dense_attn::DBuf` — the seam every
// production forward allocates from — so it holds the same path the LTX-2.5
// device suite crashes on rather than a paraphrase of it.
//
// Two test-side properties are load-bearing and neither is a style choice.
// (1) A fake backend NEVER returns a block to the C allocator: these cases
// compare pointer identity ACROSS a free, and a freed pointer is not a value you
// may reason about. (2) A fake backend is never destroyed before exit, so no
// stack or heap address is ever reused — the pool is keyed on the backend's
// identity, and an address reused by a later case would make one case's pool
// answer another case's question.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// A host-memory backend that remembers WHICH allocations are its own, so a case
// can ask the question that matters — "did this block come from THIS device?" —
// instead of inferring it from a pointer that merely happens to differ.
class TagBackend final : public Backend {
 public:
  ~TagBackend() override {
    for (void* p : owned_) std::free(p);
  }
  void* Alloc(size_t bytes) override {
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    owned_.push_back(p);
    ++allocs_;
    return p;
  }
  // Deliberately does NOT std::free: see the file header. The block stays valid
  // and stays owned; only the fact of the Free is recorded.
  void Free(void* p) override {
    freed_.push_back(p);
    ++frees_;
  }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{}; }
  bool UnifiedMemory() const override { return true; }

  bool Owns(const void* p) const {
    for (const void* q : owned_)
      if (q == p) return true;
    return false;
  }
  bool WasFreed(const void* p) const {
    for (const void* q : freed_)
      if (q == p) return true;
    return false;
  }
  int allocs() const { return allocs_; }
  int frees() const { return frees_; }

 private:
  std::vector<void*> owned_;
  std::vector<void*> freed_;
  int allocs_ = 0;
  int frees_ = 0;
};

// Process-lifetime fakes: see the file header, property (2).
TagBackend& NewBackend() {
  static std::vector<std::unique_ptr<TagBackend>> keep;
  keep.push_back(std::make_unique<TagBackend>());
  return *keep.back();
}

// Two devices of the same TYPE. `Device{type,index}` is exactly how the backend
// registry addresses discrete devices (vt/backend.h, kMaxDevicesPerType), and
// keeping the type equal keeps the platform lookup the DBuf constructor performs
// (`ResolveDevicePoolPolicy`) on a registered platform, so these cases test the
// pool and nothing else.
Queue QueueOn(int32_t index) {
  Queue q;
  q.device = Device{DeviceType::kCPU, index};
  q.handle = nullptr;
  return q;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// THE DEFECT. A block freed on device 0 must never be handed to device 1.
//
// Every case below uses its OWN size class, so no case can be decided by what an
// earlier one left in a free list — including under `--order-by=rand`.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: a block freed on one device is NEVER handed to another") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);

  // Identical shape and dtype on both devices, so both land in the SAME size
  // class. That is the whole precondition: at differing size classes the two
  // arms never trade blocks, which is why the f32 LTX-2.5 arms could not reach
  // this and the bf16 ones could.
  const std::vector<int64_t> shape{1024};  // 4096 bytes
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    on_a = x.ptr();
  }  // returned to device 0's free list here

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    on_b = y.ptr();
  }

  REQUIRE(on_a != nullptr);
  REQUIRE(on_b != nullptr);
  // The pointer identity IS the defect: before the device entered the key these
  // were the same block, and device 1 then wrote through device 0's allocation.
  CHECK(on_b != on_a);
  // ...and the stronger statement that identity check stands in for: each block
  // came out of its OWN device's allocator.
  CHECK(a.Owns(on_a));
  CHECK(b.Owns(on_b));
  CHECK_FALSE(b.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// The same ordering the other way round. The direction decides the SYMPTOM (a
// host block reaching a device forward is the silent-NaN direction; a device
// block reaching a host forward is the SIGSEGV one), so a fix that separates
// them in only one direction is not a fix.
TEST_CASE("device pool: the reverse direction is separated too") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);
  const std::vector<int64_t> shape{4096};  // 8192 bytes @ bf16

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kBF16, shape);
    on_b = y.ptr();
  }
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kBF16, shape);
    on_a = x.ptr();
  }
  CHECK(on_a != on_b);
  CHECK(b.Owns(on_b));
  CHECK(a.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// ═══════════════════════════════════════════════════════════════════════════
// ...AND THE POOL MUST STILL BE A POOL. Without the two cases below, "fixed"
// would be indistinguishable from `VT_POOL_BYPASS=1`, which also passes every
// case above and reinstates the per-op cudaMalloc/cudaFree sync storm the pool
// exists to remove. The fix has to separate the devices WITHOUT ending reuse.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: reuse on ONE device still returns the identical block") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  const std::vector<int64_t> shape{4096};  // 16384 bytes @ f32

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, shape);
    second = y.ptr();
  }
  CHECK(second == first);                   // a pool HIT...
  CHECK(a.allocs() == allocs_after_first);  // ...proven by the allocator counter
}

TEST_CASE("device pool: size-class rounding still lets nearby sizes share a block") {
  // 32,400 and 32,768 bytes round to the same class (kClassBits=4 keeps the top
  // four significant bits), which is what makes a prefill of a different token
  // count a pool hit instead of a synchronous cudaMalloc. Preserved
  // deliberately: `VT_POOL_EXACT=1` (reuse kept, rounding removed) was MEASURED
  // still red, so the rounding is not the fault and is not what this row
  // changes.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, {8100});  // 32,400 bytes
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, {8192});  // 32,768 bytes, same class
    second = y.ptr();
  }
  CHECK(second == first);
  CHECK(a.allocs() == allocs_after_first);
}
