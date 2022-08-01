// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_linear_impl.cuh"
#include <cuda_fp16.h>
#include <cub/device/device_reduce.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include <iostream>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ float RoundHalfToEven(float input) {
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

template <typename QType, typename U>
__global__ void CudaGetQuantizationParameterKernel(const U* r_max, const U* r_min, QType q_min, QType q_max, const U* input, const int64_t num_of_elements, U* sc, QType* zp) {
  U max = *r_max;
  U min = *r_min;
  min = std::min(min, U(0));
  max = std::max(max, U(0));
  *sc = max == min ? 1.0f : float(max - min) / float(q_max - q_min);
  float initial_zero_point = q_min - min / *sc;
  *zp = static_cast<QType>(RoundHalfToEven(std::max(float(q_min), std::min(float(q_max), initial_zero_point))));
}

template <typename QType, typename U>
Status CudaGetQuantizationParameter(cudaStream_t stream, const U* input, const int64_t num_of_elements, U* sc, QType* zp, const CudaKernel* cuda_kernel) {

  size_t temp_storage_bytes = 0;
  auto r_max = cuda_kernel->GetScratchBuffer<U>(sizeof(U));
  auto r_min = cuda_kernel->GetScratchBuffer<U>(sizeof(U));
  CUDA_CALL_THROW(cub::DeviceReduce::Max(nullptr, temp_storage_bytes, input, r_max.get(), num_of_elements, stream));
  auto temp_storage = cuda_kernel->GetScratchBuffer<void>(temp_storage_bytes);
  // Run max&min-reduction
  CUDA_CALL_THROW(cub::DeviceReduce::Max(temp_storage.get(), temp_storage_bytes, input, r_max.get(), num_of_elements, stream));
  CUDA_CALL_THROW(cub::DeviceReduce::Min(temp_storage.get(), temp_storage_bytes, input, r_min.get(), num_of_elements, stream));
  QType q_min = std::numeric_limits<QType>::min();
  QType q_max = std::numeric_limits<QType>::max();
  CudaGetQuantizationParameterKernel<QType, U><<<1, 1, 0, stream>>>(r_max.get(), r_min.get(), q_min, q_max, input, num_of_elements, sc, zp);
  CUDA_CALL_THROW(cudaGetLastError());
  return Status::OK();
}

template Status CudaGetQuantizationParameter<uint8_t, float>(cudaStream_t stream, const float* input, const int64_t num_of_elements, float* sc, uint8_t* zp, const CudaKernel* cuda_kernel);
//template Status CudaGetQuantizationParameter<int8_t, float>(cudaStream_t stream, const float* input, const int64_t num_of_elements, float* sc, int8_t* zp, const CudaKernel* cuda_kernel);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
