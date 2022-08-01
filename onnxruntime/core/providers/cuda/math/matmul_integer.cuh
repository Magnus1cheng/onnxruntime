// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status ReduceRowSumOnMatrixA(cudaStream_t stream, const int8_t* matrix, int32_t* row_sum, const int8_t offset, const MatMulComputeHelper& helper);
Status ReduceColSumOnMatrixB(cudaStream_t stream, const int8_t* matrix, int32_t* col_sum, const int8_t offset, const MatMulComputeHelper& helper);
Status OffsetOutput(cudaStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const int8_t a_offset,
                    const int8_t b_offset,
                    const MatMulComputeHelper& helper);

Status ReduceColSumOnMatrixB2(cudaStream_t stream, const uint8_t* matrix, int32_t* col_sum, const uint8_t offset, const MatMulComputeHelper& helper);
Status OffsetOutput2(cudaStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const uint8_t a_offset,
                    const uint8_t b_offset,
                    const MatMulComputeHelper& helper);

Status ReduceRowSumOnMatrixA3(cudaStream_t stream, const uint8_t* matrix, int32_t* row_sum, const MatMulComputeHelper& helper);
Status OffsetOutput3(cudaStream_t stream,
    const int32_t* row_sum,
    const int32_t* col_sum,
    int32_t* output,
    const uint8_t a_offset,
    const uint8_t* b_offset_ptr,
    const bool is_b_zp_per_column,
    const MatMulComputeHelper& helper);

Status Uint8WorkaroundPrecompute(cudaStream_t stream, int m, int n, int k,
    IAllocatorUniquePtr<float>& a_temp , IAllocatorUniquePtr<float>& b_temp, IAllocatorUniquePtr<float>& c_temp,
    const uint8_t* a, int lda, const uint8_t* b, int ldb, int32_t* c, int ldc, const CudaKernel* cuda_kernel);
Status Uint8WorkaroundAftercompute(cudaStream_t stream, int m, int n, float* c_temp,
    int32_t* c, int ldc);
}  // namespace cuda
}  // namespace onnxruntime
