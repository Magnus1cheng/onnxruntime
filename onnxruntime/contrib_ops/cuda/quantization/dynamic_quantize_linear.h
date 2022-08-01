// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename U = float>
class DynamicQuantizeLinear final : public CudaKernel {
 public:
  DynamicQuantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
