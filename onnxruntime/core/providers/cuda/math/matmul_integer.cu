// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.cuh"

#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2>
__global__ void CastTypeKernel(T1* dst, T2* src, int N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    dst[i] = T1(src[i]);
  }
}

template <int TPB>
__global__ void ReduceRowSumOnMatrixAKernel(const int8_t* matrix, int32_t* row_sum, const int8_t offset, int32_t K) {
  int32_t thread_data = 0;
  const int8_t* row_ptr = matrix + blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(row_ptr + i);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    row_sum[blockIdx.x] = offset * sum;
  }
}

template <int TPB>
__global__ void ReduceRowSumOnMatrixAKernel2(const uint8_t* matrix, int32_t* row_sum, const uint8_t offset, int32_t K) {
  int32_t thread_data = 0;
  const uint8_t* row_ptr = matrix + blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(row_ptr + i);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    row_sum[blockIdx.x] = sum;
  }
}

template <int TPB>
__global__ void ReduceRowSumOnMatrixAKernel3(const uint8_t* matrix, int32_t* row_sum, int32_t K) {
  int32_t thread_data = 0;
  const uint8_t* row_ptr = matrix + blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(row_ptr + i);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    row_sum[blockIdx.x] = sum;
  }
}

Status Uint8WorkaroundPrecompute(cudaStream_t stream, int m, int n, int k,
                IAllocatorUniquePtr<float>& a_temp , IAllocatorUniquePtr<float>& b_temp, IAllocatorUniquePtr<float>& c_temp,
                const uint8_t* a, int lda, const uint8_t* b, int ldb, int32_t* c, int ldc, const CudaKernel* cuda_kernel) {
  ORT_ENFORCE(cuda_kernel != nullptr, "kernel is null");
  // std::cout << "lda is:" << lda << std::endl;
  // std::cout << "ldb is:" << ldb << std::endl;
  // std::cout << "ldc is:" << ldc << std::endl;
  // std::cout << "m is:" << m << std::endl;
  // std::cout << "n is:" << n << std::endl;
  // std::cout << "k is:" << k << std::endl;

  // pad A and B to make their leading dimension be multiples of 32
  // because cublasGemmEx requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned

  //const int mask = 0x1F;
  ///int lda_aligned = lda;
  IAllocatorUniquePtr<float> a_padded;
  a_padded = cuda_kernel->GetScratchBuffer<float>(m * k * sizeof(float));
  //cudaMemcpy2DAsync(a_padded.get(), lda_aligned, a, lda, k, m, cudaMemcpyDeviceToDevice, stream);
  CastTypeKernel<float,uint8_t><<<1, GridDim::maxThreadsPerBlock, 0, stream>>>((float*)a_padded.get(), (uint8_t*)a, k*m);

  //CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass 1\n") : printf("!!!fail 1\n");
  a_temp = std::move(a_padded);
  //int ldb_aligned = ldb;
  IAllocatorUniquePtr<float> b_padded;
    b_padded = cuda_kernel->GetScratchBuffer<float>(k * n *sizeof(float));
    //cudaMemcpy2DAsync(b_padded.get(), ldb_aligned, b, ldb, n, k, cudaMemcpyDeviceToDevice, stream);
    //CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass 2\n") : printf("!!!fail 2\n");
  CastTypeKernel<float,uint8_t><<<1, GridDim::maxThreadsPerBlock, 0, stream>>>((float*)b_padded.get(), (uint8_t*)b, k*n);
  b_temp = std::move(b_padded);
  // std::cout<<"!!!!!!! stream is"<<stream<<std::endl;
  //move c
  IAllocatorUniquePtr<float> c_buff;
  c_buff = cuda_kernel->GetScratchBuffer<float>(m * n *sizeof(float));
  //cudaMemcpy2DAsync(c_buff.get(), m*sizeof(int32_t), c, m*sizeof(int32_t), m*sizeof(int32_t), n,cudaMemcpyDeviceToDevice, stream);
  CastTypeKernel<float, int32_t><<<1, GridDim::maxThreadsPerBlock, 0, stream>>>((float*)c_buff.get(), (int32_t*)c, n*m);
  c_temp = std::move(c_buff);
  // CUDA_CALL(cudaStreamSynchronize(stream));
  // CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass 2\n") : printf("!!!fail 2\n");
  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

Status Uint8WorkaroundAftercompute(cudaStream_t stream, int m, int n, float* c_temp,
                                  int32_t* c, int ldc) {
  CastTypeKernel<int32_t, float><<<1, GridDim::maxThreadsPerBlock, 0, stream>>>((int32_t*)c, (float*)c_temp, n * m);
  // CUDA_CALL(cudaStreamSynchronize(stream));
  // CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass 4\n") : printf("!!!fail 4\n");
  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
 }

Status ReduceRowSumOnMatrixA(cudaStream_t stream, const int8_t* matrix, int32_t* row_sum, const int8_t offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceRowSumOnMatrixAKernel<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.LeftOffsets()[batch],
                                                                                                                                                 row_sum + batch * helper.M(),
                                                                                                                                                 offset,
                                                                                                                                                 static_cast<int>(helper.K()));
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

Status ReduceRowSumOnMatrixA3(cudaStream_t stream, const uint8_t* matrix, int32_t* row_sum, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceRowSumOnMatrixAKernel3<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.LeftOffsets()[batch],
                                                                                                                                                 row_sum + batch * helper.M(),
                                                                                                                                                 static_cast<int>(helper.K()));
  }
  // cudaStreamSynchronize(stream);
  // CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass0\n") : printf("!!!fail0\n");

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

template <int TPB>
__global__ void ReduceColSumOnMatrixBKernel(const int8_t* matrix, int32_t* col_sum, const int8_t offset, int32_t row, int32_t col) {
  int32_t thread_data = 0;
  const int8_t* col_ptr = matrix + blockIdx.x;
  for (int i = threadIdx.x; i < row; i += TPB) {
    thread_data += *(col_ptr + i * col);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    col_sum[blockIdx.x] = offset * sum;
  }
}

template <int TPB>
__global__ void ReduceColSumOnMatrixBKernel2(const uint8_t* matrix, int32_t* col_sum, const uint8_t offset, int32_t row, int32_t col) {
  int32_t thread_data = 0;
  const uint8_t* col_ptr = matrix + blockIdx.x;
  for (int i = threadIdx.x; i < row; i += TPB) {
    thread_data += *(col_ptr + i * col);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    col_sum[blockIdx.x] = offset * sum;
  }
}

Status ReduceColSumOnMatrixB(cudaStream_t stream, const int8_t* matrix, int32_t* col_sum, const int8_t offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceColSumOnMatrixBKernel<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.N()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.RightOffsets()[batch],
                                                                                                                                                 col_sum + batch * helper.N(),
                                                                                                                                                 offset,
                                                                                                                                                 static_cast<int32_t>(helper.K()),
                                                                                                                                                 static_cast<int32_t>(helper.N()));
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

Status ReduceColSumOnMatrixB2(cudaStream_t stream, const uint8_t* matrix, int32_t* col_sum, const uint8_t offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceColSumOnMatrixBKernel2<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.N()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.RightOffsets()[batch],
                                                                                                                                                 col_sum + batch * helper.N(),
                                                                                                                                                 offset,
                                                                                                                                                 static_cast<int32_t>(helper.K()),
                                                                                                                                                 static_cast<int32_t>(helper.N()));
  }
  //CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass444\n") : printf("!!!fail555\n");
  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

__global__ void ComputeOffsetOfMatrixAB(const int32_t* row_sum,
                                        const int32_t* col_sum,
                                        int32_t* output,
                                        int32_t K_A_B,
                                        int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = K_A_B - row_sum[blockIdx.x] - col_sum[i];
  }
}

__global__ void ComputeOffsetOfMatrixAB2(const int32_t* row_sum,
                                        const int32_t* col_sum,
                                        const uint8_t* b_offset_ptr,
                                        const bool is_b_zp_per_column,
                                        int32_t* output,
                                        int32_t K_A,
                                        int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    if (is_b_zp_per_column) {
      *(output + blockIdx.x * N + i) = K_A*b_offset_ptr[i] - row_sum[blockIdx.x]*b_offset_ptr[i] - col_sum[i];
    } else {
      *(output + blockIdx.x * N + i) = K_A*b_offset_ptr[0] - row_sum[blockIdx.x]*b_offset_ptr[0] - col_sum[i];
    }
  }
}

__global__ void ComputeOffsetOfMatrixA(const int32_t* col_sum,
                                       int32_t* output,
                                       int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = -col_sum[i];
  }
}

__global__ void ComputeOffsetOfMatrixB(const int32_t* row_sum,
                                       int32_t* output,
                                       int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = -row_sum[blockIdx.x];
  }
}

__global__ void ComputeOffsetOfMatrixB2(const int32_t* row_sum,
                                       const uint8_t* b_offset_ptr,
                                       const bool is_b_zp_per_column,
                                       int32_t* output,
                                       int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    if (is_b_zp_per_column) {
      *(output + blockIdx.x * N + i) = -row_sum[blockIdx.x]*b_offset_ptr[i];
    } else {
      *(output + blockIdx.x * N + i) = -row_sum[blockIdx.x]*b_offset_ptr[0];
    }
  }
}

Status OffsetOutput(cudaStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const int8_t a_offset,
                    const int8_t b_offset,
                    const MatMulComputeHelper& helper) {
  if (a_offset && b_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixAB<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          col_sum + batch * helper.N(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.K()) * a_offset * b_offset,
          static_cast<int32_t>(helper.N()));
    }
  } else if (a_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixA<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          col_sum + batch * helper.N(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.N()));
    }
  } else if (b_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixB<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.N()));
    }
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

Status OffsetOutput3(cudaStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const uint8_t a_offset,
                    const uint8_t* b_offset_ptr,
                    const bool is_b_zp_per_column,
                    const MatMulComputeHelper& helper) {
  if (a_offset && b_offset_ptr!= nullptr ) { //is_b_zp_per_column
    //std::cout<< "!!!has a_offset" << std::endl;
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixAB2<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          col_sum + batch * helper.N(),
          b_offset_ptr,
          is_b_zp_per_column,
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.K()) * a_offset,
          static_cast<int32_t>(helper.N()));
    }
  } else if (b_offset_ptr!= nullptr) {
    std::cout<< "!!!not hav a_offset" << std::endl;
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixB2<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          b_offset_ptr,
          is_b_zp_per_column,
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.N()));
    }
  } else if (a_offset) {
    std::cout<< "!!!b_offset_ptr is null ptr" << std::endl;
  }
  // cudaStreamSynchronize(stream);
  // CUDA_CALL(cudaPeekAtLastError()) ? printf("!!!pass1\n") : printf("!!!fail1\n");

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

}  // namespace cuda
}  // namespace onnxruntime
