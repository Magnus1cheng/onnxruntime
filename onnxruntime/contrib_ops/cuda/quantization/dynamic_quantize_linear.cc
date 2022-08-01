// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_linear.h"
#include "dynamic_quantize_linear_impl.cuh"
#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include <iostream>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_Q_KERNEL_TYPED(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      DynamicQuantizeLinear,                                           \
      kOnnxDomain,                                                     \
      11,                                                              \
      T,                                                               \
      kCudaExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),     \
      DynamicQuantizeLinear<T, float>);

//REGISTER_Q_KERNEL_TYPED(int8_t)
REGISTER_Q_KERNEL_TYPED(uint8_t)

// formula is Y = X / Scale + ZeroPoint
template <typename T, typename U>
Status DynamicQuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<T>::MappedType CudaT;

  auto& x = *ctx->Input<Tensor>(0);
  const auto num_of_elements = x.Shape().Size();
  const CudaU* x_data = reinterpret_cast<const CudaU*>(x.template Data<U>());

  auto& y = *ctx->Output(0, x.Shape());
  TensorShape shape({1});
  auto& y_scale = *ctx->Output(1, shape);
  auto& y_zeropoint = *ctx->Output(2, shape);

  const CudaU* input = reinterpret_cast<const CudaU*>(x.template Data<U>());

  CudaU* output_scale = reinterpret_cast<CudaU*>(y_scale.template MutableData<U>());
  CudaT* output_zp = reinterpret_cast<CudaT*>(y_zeropoint.template MutableData<T>());
  ORT_RETURN_IF_ERROR(CudaGetQuantizationParameter(Stream(), x_data, num_of_elements, output_scale, output_zp, this));

  // quantize the data
  CudaT* output = reinterpret_cast<CudaT*>(y.template MutableData<T>());
  ORT_RETURN_IF_ERROR(CudaQuantizeLinear(Stream(), input, output, output_scale, output_zp, num_of_elements));
  //CUDA_CALL(cudaGetLastError());
  return Status::OK();
}

// explicit instantation for DynamicQuantizeLinear::ComputeInternal
#define SPECIALIZED_Q_COMPUTE(T, U)                                                         \
  template Status DynamicQuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const; \

//SPECIALIZED_Q_COMPUTE(int8_t, float)
SPECIALIZED_Q_COMPUTE(uint8_t, float)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
