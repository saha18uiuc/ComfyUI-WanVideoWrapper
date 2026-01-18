/*
 * Fused CUDA Kernels for WanVideo
 * 
 * These kernels reduce memory bandwidth by fusing operations:
 * - fused_silu_mul: 25% less memory traffic
 * - fused_rmsnorm: 50% less memory traffic
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Fused SiLU * Gate (SwiGLU activation)
// ============================================================================

template <typename scalar_t>
__global__ void fused_silu_mul_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ out,
    const int64_t n
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x_val = static_cast<float>(x[idx]);
        float gate_val = static_cast<float>(gate[idx]);
        
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float silu_x = x_val / (1.0f + expf(-x_val));
        
        out[idx] = static_cast<scalar_t>(silu_x * gate_val);
    }
}

torch::Tensor fused_silu_mul_cuda(torch::Tensor x, torch::Tensor gate) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA tensor");
    TORCH_CHECK(x.sizes() == gate.sizes(), "x and gate must have same shape");
    
    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_silu_mul_cuda", ([&] {
        fused_silu_mul_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n
        );
    }));
    
    return out;
}

// ============================================================================
// Fused RMSNorm
// ============================================================================

template <typename scalar_t>
__global__ void fused_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    const int64_t rows,
    const int64_t cols,
    const float eps
) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    
    const scalar_t* x_row = x + row * cols;
    scalar_t* out_row = out + row * cols;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]);
        sum_sq += val * val;
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // Block reduction using shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / warpSize) {
        sum_sq = shared[lane];
    } else {
        sum_sq = 0.0f;
    }
    
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast result
    __shared__ float rstd;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / static_cast<float>(cols);
        rstd = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    
    // Normalize and apply weight
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<scalar_t>(val * rstd * w);
    }
}

torch::Tensor fused_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    
    const int64_t cols = x.size(-1);
    const int64_t rows = x.numel() / cols;
    
    auto out = torch::empty_like(x);
    
    // Use 256 threads per block
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_rmsnorm_cuda", ([&] {
        fused_rmsnorm_kernel<scalar_t><<<rows, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(eps)
        );
    }));
    
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul", &fused_silu_mul_cuda, "Fused SiLU * gate (CUDA)");
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, "Fused RMSNorm (CUDA)");
}
