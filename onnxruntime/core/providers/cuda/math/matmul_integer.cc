// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.h"
#include "matmul_integer.cuh"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/common.h"
#include <iostream>

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    uint8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, uint8_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);

template <>
Status MatMulInteger<int8_t, int8_t>::ComputeInternal(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const int8_t* a_ptr = a->template Data<int8_t>();
  const int8_t* b_ptr = b->template Data<int8_t>();
  int32_t* output_ptr = Y->template MutableData<int32_t>();

  // validate zero points
  int8_t a_offset = 0;
  int8_t b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(a_zero_point->template Data<int8_t>());
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = *(b_zero_point->template Data<int8_t>());
  }
  // offset output c[i,j] to
  // k*a_offset*b_offset -
  // b_offset * (a[i,0] + a[i,1] ...+a[i,k]) -
  // a_offset * (b[0,j] + b[1,j] ... + b[k,j])
  // ReduceRowSumOnMatrixA computes the b_offset * (a[i,0] + a[i,1] ...+a[i,k]) part
  // ReduceColSumOnMatrixB computes the a_offset * (b[0,j] + b[1,j] ... + b[k,j]) part
  // OffsetOutput computes gets the final result
  IAllocatorUniquePtr<int32_t> a_row_buf;
  if (b_offset != 0) {
    a_row_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.N());
    ORT_RETURN_IF_ERROR(ReduceRowSumOnMatrixA(Stream(), a_ptr, a_row_buf.get(), b_offset, helper));
  }

  IAllocatorUniquePtr<int32_t> b_col_buf;
  if (a_offset != 0) {
    b_col_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.M());
    ORT_RETURN_IF_ERROR(ReduceColSumOnMatrixB(Stream(), b_ptr, b_col_buf.get(), a_offset, helper));
  }

  int alpha = 1;
  int beta = 0;
  if (a_offset != 0 || b_offset != 0) {
    ORT_RETURN_IF_ERROR(OffsetOutput(Stream(),
                                     a_row_buf.get(),
                                     b_col_buf.get(),
                                     output_ptr,
                                     a_offset,
                                     b_offset,
                                     helper));
    beta = 1;
  }

  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ORT_RETURN_IF_ERROR(GemmInt8(static_cast<int>(helper.M()),
                                 static_cast<int>(helper.N()),
                                 static_cast<int>(helper.K()),
                                 alpha,
                                 beta,
                                 a_ptr + helper.LeftOffsets()[batch],
                                 static_cast<int>(helper.K()),
                                 b_ptr + helper.RightOffsets()[batch],
                                 static_cast<int>(helper.N()),
                                 output_ptr + helper.OutputOffsets()[batch],
                                 static_cast<int>(helper.N()),
                                 this));
  }

  return Status::OK();
}
//implement for int8_t, uint8_t
template <>
Status MatMulInteger<uint8_t, uint8_t>::ComputeInternal(OpKernelContext* ctx) const {
  //std::cout<<"!!!!!!!!!!!!!!!! in MatMulInteger uint8" << std::endl;
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const uint8_t* a_ptr =  static_cast<const uint8_t*>(a->DataRaw());
  const uint8_t* b_ptr =  static_cast<const uint8_t*>(b->DataRaw());
  // std::cout<< "!!!!!!!a info:"<< a->Location().ToString() << std::endl;
  // std::cout<< "!!!!!!!b info:"<< b->Location().ToString() << std::endl;
  int32_t* output_ptr = Y->template MutableData<int32_t>();

  // offset output c[i,j] to
  // k*a_offset*b_offset[j] -
  // b_offset[j] * (a[i,0] + a[i,1] ...+a[i,k]) -
  // a_offset * (b[0,j] + b[1,j] ... + b[k,j])
  // ReduceRowSumOnMatrixA computes the b_offset * (a[i,0] + a[i,1] ...+a[i,k]) part
  // ReduceColSumOnMatrixB computes the a_offset * (b[0,j] + b[1,j] ... + b[k,j]) part
  // OffsetOutput computes gets the final result
  // validate zero points
  uint8_t a_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    //a_offset_ptr = static_cast<const uint8_t*>(a_zero_point->DataRaw());
    // std::cout<< "!!gpu: a_zero_point DataType():" << DataTypeImpl::ToString(a_zero_point->DataType()) << std::endl;
  } else {
    std::cout<< "!!!! not have a zero point" << std::endl;
  }
  IAllocatorUniquePtr<int32_t> a_row_buf;
  bool is_b_zp_per_column = false;
  const uint8_t* b_offset_ptr = nullptr;
  if (has_b_zero_point_) {
      auto b_zero_point = ctx->Input<Tensor>(3);
      b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
      // std::cout<< "!!b shape is" << b->Shape() << std::endl;
      ORT_ENFORCE(IsBQuantParamSupported(b_zero_point->Shape(),b->Shape()),
                  "MatmulInteger : B zero point is not valid");
      is_b_zp_per_column = !IsScalarOr1ElementVector(b_zero_point);
      //auto b_offset_buf = GetScratchBuffer<uint8_t>(b_zero_point->SizeInBytes());
      //CUDA_CALL(cudaMemcpyAsync(b_offset_buf.get(), b_zero_point->DataRaw(), b_zero_point->SizeInBytes(), cudaMemcpyHostToDevice, Stream()));
      //b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
      //b_offset_ptr =b_offset_buf.get();
      // std::cout<< "!!!!!!!b_zero_point info:"<< b_zero_point->Location().ToString() << std::endl;
      a_row_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.N());
      ORT_RETURN_IF_ERROR(ReduceRowSumOnMatrixA3(Stream(), a_ptr, a_row_buf.get(), helper));
      //  std::cout<< "!!gpu: b_zp DataType():" <<  DataTypeImpl::ToString(b_zero_point->DataType()) << std::endl;
  } else {
    std::cout<< "!!!! not have b zero point" << std::endl;
  }

  IAllocatorUniquePtr<int32_t> b_col_buf;
  if (a_offset != 0) {
    b_col_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.M());
    ORT_RETURN_IF_ERROR(ReduceColSumOnMatrixB2(Stream(), b_ptr, b_col_buf.get(), a_offset, helper));
  }

  float alpha = 1.f;
  float beta = 0.f;
  if (a_offset != 0) {
    ORT_RETURN_IF_ERROR(OffsetOutput3(Stream(),
                                     a_row_buf.get(),
                                     b_col_buf.get(),
                                     output_ptr,
                                     a_offset,
                                     b_offset_ptr,
                                     is_b_zp_per_column,
                                     helper));
    beta = 1.f;
  }

  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    IAllocatorUniquePtr<float> a_tmp_ptr;
    IAllocatorUniquePtr<float> b_tmp_ptr;
    IAllocatorUniquePtr<float> c_tmp_ptr;
    // float* a_tmp_ptr = NULL;
    // float* b_tmp_ptr = NULL;
    // float* c_tmp_ptr = NULL;
    //check 调用是否有问题 MNK参数
    ORT_RETURN_IF_ERROR(Uint8WorkaroundPrecompute(Stream(),
                                 static_cast<int>(helper.M()),
                                 static_cast<int>(helper.N()),
                                 static_cast<int>(helper.K()),
                                 a_tmp_ptr,
                                 b_tmp_ptr,
                                 c_tmp_ptr,
                                 a_ptr + helper.LeftOffsets()[batch],
                                 static_cast<int>(helper.K()),
                                 b_ptr + helper.RightOffsets()[batch],
                                 static_cast<int>(helper.N()),
                                 output_ptr + helper.OutputOffsets()[batch],
                                 static_cast<int>(helper.N()),
                                 this));
    // if(a_tmp_ptr!=nullptr) {
    //   std::cout<<"!!!a_tmp_ptr!=nullptr"<<std::endl;
    //   std::cout<< a_tmp_ptr.get()<<std::endl;
    // } else {
    //   std::cout<<"!!!a_tmp_ptr==nullptr"<<std::endl;
    // }
    ORT_RETURN_IF_ERROR(GemmUInt8(static_cast<int>(helper.M()),
                                 static_cast<int>(helper.N()),
                                 static_cast<int>(helper.K()),
                                 alpha,
                                 beta,
                                 a_tmp_ptr.get(),
                                 static_cast<int>(helper.K()),
                                 b_tmp_ptr.get(),
                                 static_cast<int>(helper.N()),
                                 c_tmp_ptr.get(),
                                 //output_ptr + helper.OutputOffsets()[batch],
                                 static_cast<int>(helper.N()),
                                 this));
    ORT_RETURN_IF_ERROR(Uint8WorkaroundAftercompute(Stream(),
                                 static_cast<int>(helper.M()),
                                 static_cast<int>(helper.N()),
                                 c_tmp_ptr.get(),
                                 output_ptr + helper.OutputOffsets()[batch],
                                 static_cast<int>(helper.N())));

  }
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
