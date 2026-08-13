# SYCL Backend Spec

## Scope
Add a SYCL backend to vllm.cpp mirroring the ROCm backend approach.

## Upstream anchors
vLLM does not have a SYCL backend. Reference implementation is Intel vllm-xpu-kernels and SYCL-TLA.

## Design
* Add DeviceType::kSYCL to include/vt/device.h
* Add vt::sycl::Backend implementation in src/vt/sycl/
* Add Platform seam src/vllm/platforms/sycl.cpp
* CMake option VLLM_CPP_SYCL ON/OFF/AUTO
* W0 skeleton: platform registration, backend probe for unified memory, no kernels yet.

## Tests
* test_rocm_backend equivalent for SYCL
* Compile guard

## Gates
* Configure with -DVLLM_CPP_SYCL=ON succeeds
* Platform registers

## Outcome
W0 skeleton created.
