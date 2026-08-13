// SYCL platform seam
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
namespace vllm::platforms {
class SyclPlatform final : public Platform {
public:
  DeviceType device_type() const override { return DeviceType::kSYCL; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kSYCL); }
};
}
