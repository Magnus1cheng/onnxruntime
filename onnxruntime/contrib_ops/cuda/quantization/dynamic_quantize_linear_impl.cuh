// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename QType, typename U>
Status CudaGetQuantizationParameter(cudaStream_t stream, const U* input, const int64_t num_of_elements, U* sc, QType* zp, const CudaKernel* cuda_kernel);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
