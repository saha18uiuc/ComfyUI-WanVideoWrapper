/*
 * Fused CUDA Kernels for WanVideo
 * 
 * These kernels fuse multiple operations into single GPU passes,
 * reducing memory bandwidth which is the main bottleneck on modern GPUs.
 * 
 * Compilation: Automatically compiled by torch.utils.cpp_extension.load()
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Fused SiLU * Gate (SwiGLU activation)
// ============================================================================
// 
// Standard PyTorch: silu(x) * gate
//   - Read x (N elements)
//   - Compute silu, write intermediate (N elements)  
//   - Read intermediate + gate (2N elements)
//   - Write output (N elements)
//   Total: 4N memory operations
//
// Fused kernel:
//   - Read x + gate (2N elements)
//   - Write output (N elements)
//   Total: 3N memory operations = 25% less memory traffic
//
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
        // Load x and gate in single memory transaction (coalesced)
        float x_val = static_cast<float>(x[idx]);
        float gate_val = static_cast<float>(gate[idx]);
        
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float silu_x = x_val / (1.0f + expf(-x_val));
        
        // Fused output
        out[idx] = static_cast<scalar_t>(silu_x * gate_val);
    }
}

// Vectorized version for fp16 - processes 2 elements at a time
__global__ void fused_silu_mul_kernel_half2(
    const __half2* __restrict__ x,
    const __half2* __restrict__ gate,
    __half2* __restrict__ out,
    const int64_t n  // n is number of half2 elements (original_n / 2)
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        __half2 x_val = x[idx];
        __half2 gate_val = gate[idx];
        
        // Convert to float2 for computation
        float2 x_f = __half22float2(x_val);
        float2 gate_f = __half22float2(gate_val);
        
        // SiLU for both elements
        float2 silu_f;
        silu_f.x = x_f.x / (1.0f + expf(-x_f.x));
        silu_f.y = x_f.y / (1.0f + expf(-x_f.y));
        
        // Multiply with gate
        float2 out_f;
        out_f.x = silu_f.x * gate_f.x;
        out_f.y = silu_f.y * gate_f.y;
        
        out[idx] = __float22half2_rn(out_f);
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
    
    // Use vectorized kernel for fp16 when possible
    if (x.scalar_type() == torch::kHalf && n % 2 == 0) {
        const int64_t n_half2 = n / 2;
        const int blocks_half2 = (n_half2 + threads - 1) / threads;
        
        fused_silu_mul_kernel_half2<<<blocks_half2, threads>>>(
            reinterpret_cast<const __half2*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half2*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
            n_half2
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_silu_mul_cuda", ([&] {
            fused_silu_mul_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n
            );
        }));
    }
    
    return out;
}


// ============================================================================
// Fused RMSNorm
// ============================================================================
//
// Standard PyTorch: x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
//   - pow(2): Read x, write x^2 (2N ops)
//   - mean: Read x^2, write mean (N + B ops)
//   - rsqrt: Read mean, write rsqrt (2B ops)
//   - mul: Read x, rsqrt, write (2N + B ops)
//   - mul weight: Read, weight, write (2N + D ops)
//   Total: ~8N memory operations
//
// Fused kernel (2-pass):
//   - Pass 1: Read x, compute partial sums (N ops)
//   - Pass 2: Read x, weight, write output (3N ops)
//   Total: 4N memory operations = 50% less memory traffic
//
// ============================================================================

template <typename scalar_t, int BLOCK_SIZE>
__global__ void fused_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    const int64_t hidden_dim,
    const float eps
) {
    // Each block processes one row
    const int64_t row = blockIdx.x;
    const scalar_t* x_row = x + row * hidden_dim;
    scalar_t* out_row = out + row * hidden_dim;
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    
    // Step 1: Compute sum of squares
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE) {
        float val = static_cast<float>(x_row[i]);
        thread_sum += val * val;
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Compute normalization factor
    float mean_sq = shared_sum[0] / static_cast<float>(hidden_dim);
    float rstd = rsqrtf(mean_sq + eps);
    
    // Step 2: Normalize and apply weight
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE) {
        float val = static_cast<float>(x_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<scalar_t>(val * rstd * w);
    }
}


torch::Tensor fused_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    
    const int64_t hidden_dim = x.size(-1);
    const int64_t num_rows = x.numel() / hidden_dim;
    
    auto out = torch::empty_like(x);
    
    // Choose block size based on hidden dimension
    const int BLOCK_SIZE = 256;  // Good for most cases
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_rmsnorm_cuda", ([&] {
        fused_rmsnorm_kernel<scalar_t, BLOCK_SIZE><<<num_rows, BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            hidden_dim,
            eps
        );
    }));
    
    return out;
}


// ============================================================================
// Fused Add + RMSNorm (for residual connections)
// ============================================================================
// 
// Pattern: rmsnorm(x + residual)
// Fuses the add into the norm to save one full tensor read/write
//
// ============================================================================

template <typename scalar_t, int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    const int64_t hidden_dim,
    const float eps
) {
    const int64_t row = blockIdx.x;
    const scalar_t* x_row = x + row * hidden_dim;
    const scalar_t* res_row = residual + row * hidden_dim;
    scalar_t* out_row = out + row * hidden_dim;
    
    __shared__ float shared_sum[BLOCK_SIZE];
    
    // Compute sum of squares of (x + residual)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE) {
        float val = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        thread_sum += val * val;
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean_sq = shared_sum[0] / static_cast<float>(hidden_dim);
    float rstd = rsqrtf(mean_sq + eps);
    
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE) {
        float val = static_cast<float>(x_row[i]) + static_cast<float>(res_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<scalar_t>(val * rstd * w);
    }
}


torch::Tensor fused_add_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    float eps
) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda());
    TORCH_CHECK(x.sizes() == residual.sizes());
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous());
    
    const int64_t hidden_dim = x.size(-1);
    const int64_t num_rows = x.numel() / hidden_dim;
    
    auto out = torch::empty_like(x);
    const int BLOCK_SIZE = 256;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_add_rmsnorm_cuda", ([&] {
        fused_add_rmsnorm_kernel<scalar_t, BLOCK_SIZE><<<num_rows, BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            hidden_dim,
            eps
        );
    }));
    
    return out;
}


// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul", &fused_silu_mul_cuda, "Fused SiLU * gate (CUDA)");
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, "Fused RMSNorm (CUDA)");
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm_cuda, "Fused Add + RMSNorm (CUDA)");
}
