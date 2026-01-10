import os
import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 64
BLOCK_R = 32


@triton.jit
def _grouped_lora_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    batch,
    in_features,
    out_features,
    rank,
    num_loras,
    stride_x_row,
    stride_x_col,
    stride_a_lora,
    stride_a_rank,
    stride_a_in,
    stride_b_lora,
    stride_b_out,
    stride_b_rank,
    stride_out_row,
    stride_out_col,
    *,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < batch
    mask_n = offs_n < out_features

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for lora_idx in range(num_loras):
        scale = tl.load(scale_ptr + lora_idx)
        for rk in range(0, rank, BLOCK_R):
            offs_r = rk + tl.arange(0, BLOCK_R)
            mask_r = offs_r < rank
            tmp = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

            for k in range(0, in_features, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < in_features
                x = tl.load(
                    x_ptr
                    + (offs_m[:, None] * stride_x_row + offs_k[None, :] * stride_x_col),
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                a_block = tl.load(
                    a_ptr
                    + lora_idx * stride_a_lora
                    + offs_r[:, None] * stride_a_rank
                    + offs_k[None, :] * stride_a_in,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                tmp += tl.dot(x, tl.trans(a_block))

            b_block = tl.load(
                b_ptr
                + lora_idx * stride_b_lora
                + offs_n[:, None] * stride_b_out
                + offs_r[None, :] * stride_b_rank,
                mask=mask_n[:, None] & mask_r[None, :],
                other=0.0,
            )

            acc += scale * tl.dot(tmp, tl.trans(b_block))

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_row + offs_n[None, :] * stride_out_col,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def grouped_lora_available():
    if not _HAS_TRITON:
        return False
    return True


def grouped_lora_forward(x, a, b, scales):
    """
    Compute fused LoRA contribution for a batch of inputs.

    Args:
        x (Tensor): [batch, in_features], fp16/bf16/fp32
        a (Tensor): [num_loras, rank, in_features]
        b (Tensor): [num_loras, out_features, rank]
        scales (Tensor): [num_loras], fp32
    """
    if not grouped_lora_available():
        raise RuntimeError("Triton is required for grouped LoRA kernels.")
    if not x.is_cuda:
        raise RuntimeError("Grouped LoRA kernel requires CUDA tensors.")
    if x.numel() == 0:
        return torch.zeros(x.shape[0], b.shape[1], device=x.device, dtype=x.dtype)
    if not (a.is_cuda and b.is_cuda and scales.is_cuda):
        raise RuntimeError("Grouped LoRA kernel expects CUDA tensors.")

    batch, in_features = x.shape
    num_loras, rank, _ = a.shape
    out_features = b.shape[1]

    out = torch.zeros((batch, out_features), device=x.device, dtype=torch.float32)
    grid = (
        triton.cdiv(batch, BLOCK_M),
        triton.cdiv(out_features, BLOCK_N),
    )

    _grouped_lora_kernel[grid](
        x,
        a,
        b,
        scales,
        out,
        batch,
        in_features,
        out_features,
        rank,
        num_loras,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
    )
    return out.to(x.dtype)
