// SYCL backend skeleton
#pragma once
#include "vt/backend.h"
namespace vt::sycl {
class Backend : public vt::Backend {
public:
  int DeviceCapabilityMajor() const override { return 0; }
  int DeviceCapabilityMinor() const override { return 0; }
  bool UnifiedMemory() const override { return false; }
};
}
