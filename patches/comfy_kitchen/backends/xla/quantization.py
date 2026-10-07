"""XLA (TPU) INT8 linear.

Semantics match the tpu-inference reference ``xla_quantized_matmul``
(``tpu_inference/layers/common/linear.py``) exactly on the case ComfyUI's
``int8_linear`` actually hits (see ``TensorWiseINT8Layout``): a scalar or
per-output-channel (1-D) weight scale plus dynamic per-row (per-token)
activation quantization. The contract, in order:

1. quantize the activation to INT8 with a per-row scale (abs-max / 127);
2. integer matmul, INT32 accumulation (INT8 operands);
3. multiply the integer result by the per-row activation scale — AFTER the
   integer matmul;
4. multiply by the per-output-channel weight scale;
5. cast to the output dtype, add bias, reshape, apply residual.

Getting the scale placement in step 3/4 wrong produces plausible garbage
rather than an error, so the order is load-bearing.

The matmul goes through ``torch._int_mm`` (int8 x int8 -> int32), the same
primitive the eager backend's INT8 path falls back to; on torch_xla it
lowers to an integer dot. If the lowering rejects a shape (the only
failure mode we cannot rule out on TPU from a CPU-only probe), a
float32-dequant matmul takes over: exact for K <= 4096, a numerics
fallback (not an equivalence) beyond that. The function itself is pure
torch ops, so it is testable on cpu/meta without initializing any PJRT
client.
"""

from __future__ import annotations

import logging

import torch

from comfy_kitchen.backends._activations import (
    apply_input_act as _apply_input_act,
)
from comfy_kitchen.backends._activations import (
    apply_residual as _apply_residual,
)
from comfy_kitchen.backends.eager.quantization import (
    quantize_int8_rowwise as _quantize_int8_rowwise,
)
from comfy_kitchen.tensor.int8_utils import (
    _build_hadamard,
    _rotate_activation,
)

logger = logging.getLogger("comfy_kitchen.xla")

_DEQUANT_FALLBACK_WARNED = False


def _int8_matmul_int32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for INT8 operands with INT32 accumulation.

    Primary: ``torch._int_mm`` — exact. Fallback (only when the op is
    missing or the lowering rejects the shape): dequantize both operands
    to float32 and matmul; int8 products and sums up to K=4096 stay
    exact in float32 (sum bound 4096 * 127 * 127 < 2**24), so the
    fallback is exact for the shapes this model uses and degrades to a
    float-precision path only beyond that.
    """
    if hasattr(torch, "_int_mm"):
        try:
            return torch._int_mm(a, b)
        except Exception:
            pass
    global _DEQUANT_FALLBACK_WARNED
    if not _DEQUANT_FALLBACK_WARNED:
        logger.warning(
            "xla int8_linear: torch._int_mm unavailable/rejected shape %s x %s; "
            "using float32 dequant matmul (exact for K <= 4096)",
            tuple(a.shape),
            tuple(b.shape),
        )
        _DEQUANT_FALLBACK_WARNED = True
    return (a.to(torch.float32) @ b.to(torch.float32)).to(torch.int32)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 linear layer for XLA devices (TPU), ConvRot variant included.

    Args mirror the public ``comfy_kitchen.int8_linear`` contract:
        x: [..., K] activation, standard float dtype.
        weight: [N, K] INT8 weight (output channels x input channels).
        weight_scale: scalar or per-output-channel [N] float scale.
        bias: optional [N] bias, added in out_dtype after scaling.
        out_dtype: output dtype (callers pass x.dtype).
        convrot: rotate x through grouped Hadamard before quantization
            (weights were rotated the same way at quantize time).
        residual/residual_scale: result becomes
            ``residual + residual_scale * linear(x)``.
    """
    if out_dtype is None:
        out_dtype = torch.bfloat16
    x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Dynamic per-row (per-token) activation quantization to INT8.
    x_8, x_scale = _quantize_int8_rowwise(x_2d)

    # weight is [N, K]; the reference contracts x's last dim against the
    # weight's last dim, so the matmul sees weight transposed to [K, N].
    acc = _int8_matmul_int32(x_8, weight.T.contiguous())

    # Reference scale order: per-row activation scale AFTER the integer
    # matmul, then per-output-channel weight scale, then output dtype.
    out = acc.to(torch.float32) * x_scale
    out = out * weight_scale.reshape(1, -1)
    out = out.to(out_dtype)

    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out_dtype).reshape(1, -1)

    out = out.reshape(*orig_shape[:-1], weight.shape[0])
    return _apply_residual(out, residual, residual_scale)
