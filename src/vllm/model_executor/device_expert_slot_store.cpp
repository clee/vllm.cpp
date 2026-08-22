// ENG-EXPERT-STREAM-DEVICE W1 (issue #1124). See device_expert_slot_store.h for
// why the destination is device memory, why the fill is a staging bounce, and
// why there is exactly one staging buffer.
#include "vllm/model_executor/device_expert_slot_store.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace vllm {

DeviceExpertSlotStore::DeviceExpertSlotStore(vt::Backend& backend,
                                             int32_t slots, size_t slot_bytes)
    : b_(backend), slots_(slots), slot_bytes_(slot_bytes) {
  // Every refusal below happens BEFORE the queue is created, deliberately: a
  // constructor that throws has no destructor run, so anything acquired above
  // the throw leaks. Nothing is acquired until the budget is known good.
  if (slots <= 0) {
    throw std::invalid_argument(
        "DeviceExpertSlotStore: slot count must be > 0");
  }
  if (slot_bytes == 0) {
    throw std::invalid_argument("DeviceExpertSlotStore: slot bytes must be > 0");
  }
  // The host store's arena is a `std::vector`, whose own length check catches
  // this; a raw `Alloc` has no such backstop, and an arena that wrapped would
  // hand out in-range slot pointers past its end — a silent overwrite rather
  // than a refusal.
  if (slot_bytes > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(slots)) {
    throw std::invalid_argument(
        "DeviceExpertSlotStore: " + std::to_string(slots) + " slots of " +
        std::to_string(slot_bytes) + " bytes overflows size_t");
  }

  const size_t total = static_cast<size_t>(slots) * slot_bytes;
  q_ = b_.CreateQueue();
  arena_ = static_cast<uint8_t*>(b_.Alloc(total));
  if (arena_ == nullptr) {
    b_.DestroyQueue(q_);
    throw std::runtime_error("DeviceExpertSlotStore: device allocation of " +
                             std::to_string(total) + " bytes failed");
  }
  // Page-locked, because this buffer is the source of every H2D the lane
  // issues: on CUDA a copy from pageable memory stages through a driver bounce
  // of its own, which is the one thing this design must not pay twice. The base
  // implementation returns ordinary host memory, which is correct on a backend
  // where the distinction does not exist.
  staging_ = static_cast<uint8_t*>(b_.AllocPinned(slot_bytes));
  if (staging_ == nullptr) {
    b_.Free(arena_);
    b_.DestroyQueue(q_);
    throw std::runtime_error("DeviceExpertSlotStore: pinned staging allocation "
                             "of " + std::to_string(slot_bytes) +
                             " bytes failed");
  }
}

DeviceExpertSlotStore::~DeviceExpertSlotStore() {
  if (staging_ != nullptr) b_.FreePinned(staging_);
  if (arena_ != nullptr) b_.Free(arena_);
  b_.DestroyQueue(q_);
}

uint8_t* DeviceExpertSlotStore::SlotPtr(int32_t slot) const {
  if (slot < 0 || slot >= slots_) {
    throw std::out_of_range("DeviceExpertSlotStore: slot " +
                            std::to_string(slot) + " out of range");
  }
  return arena_ + static_cast<size_t>(slot) * slot_bytes_;
}

void DeviceExpertSlotStore::WriteSlot(int32_t slot, const uint8_t* src,
                                      size_t bytes) {
  uint8_t* dst = SlotPtr(slot);  // bounds first, as the host store does
  if (bytes > slot_bytes_) {
    throw std::invalid_argument("DeviceExpertSlotStore: write of " +
                                std::to_string(bytes) +
                                " bytes exceeds the slot");
  }
  if (src == nullptr && bytes > 0) {
    throw std::invalid_argument("DeviceExpertSlotStore: null source");
  }
  if (bytes == 0) return;
  b_.Copy(q_, dst, src, bytes);
  // Synchronous by contract: the caller binds this slot to a GEMM as soon as
  // the streamer returns, and `Copy` is asynchronous on CUDA.
  b_.Synchronize(q_);
}

uint8_t* DeviceExpertSlotStore::SlotForWrite(int32_t slot) {
  if (slot < 0 || slot >= slots_) {
    throw std::out_of_range("DeviceExpertSlotStore: slot " +
                            std::to_string(slot) + " out of range");
  }
  staged_slot_ = slot;
  return staging_;
}

void DeviceExpertSlotStore::CommitSlot(int32_t slot, size_t bytes) {
  uint8_t* dst = SlotPtr(slot);
  if (bytes > slot_bytes_) {
    throw std::invalid_argument("DeviceExpertSlotStore: commit of " +
                                std::to_string(bytes) +
                                " bytes exceeds the slot");
  }
  if (staged_slot_ != slot) {
    // With one staging buffer this is not a bookkeeping slip: the bytes in
    // staging belong to whichever slot asked for it last, so committing them
    // here would file one expert's weights under another expert's key. The
    // cache would then report a HIT for a key whose slot holds the wrong
    // expert, and the GEMM would multiply it without a symptom.
    throw std::logic_error(
        "DeviceExpertSlotStore: commit of slot " + std::to_string(slot) +
        " but SlotForWrite last staged " + std::to_string(staged_slot_));
  }
  staged_slot_ = -1;
  if (bytes == 0) return;
  b_.Copy(q_, dst, staging_, bytes);
  b_.Synchronize(q_);
}

uint8_t* DeviceExpertSlotStore::SlotForRead(int32_t slot) {
  return SlotPtr(slot);
}

}  // namespace vllm
