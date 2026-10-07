# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XLA (TPU) backend.

Optional like ascend/triton: importing :mod:`comfy_kitchen` must not require
torch_xla on CPU/CUDA-only installs, so the backend registers only when
``import torch_xla`` succeeds and marks itself unavailable otherwise.

Device scope is deliberately narrow: the capability declares
``default_devices=frozenset({"xla"})``, so a tensor on any other device type
fails this backend's constraint validation and dispatch continues down the
priority list. That is what makes XLA dispatch deterministic: without it,
xla-device tensors reach the eager backend only through the eager ``"*"``
wildcard, which is incidental rather than a decision.

Priority: the registry order is ``["ascend", "cuda", "triton", "xla",
"eager"]`` — xla sits after the accelerator backends it never overlaps
(their device sets exclude "xla") and before eager, the only backend whose
wildcard would otherwise swallow xla tensors.
"""

from __future__ import annotations

import torch

from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry

from .quantization import int8_linear

__all__ = [
    "int8_linear",
]

_XLA_AVAILABLE = True
_XLA_ERROR: str | None = None

try:
    import torch_xla  # noqa: F401
except ImportError as exc:
    _XLA_AVAILABLE = False
    _XLA_ERROR = f"torch-xla is not installed: {exc}"
except Exception as exc:  # a broken PJRT env must not kill the package import
    _XLA_AVAILABLE = False
    _XLA_ERROR = f"torch-xla initialization failed: {exc}"


def _build_constraints() -> dict:
    xla_devices = frozenset({"xla"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    return {
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "weight_scale": ParamConstraint(dtypes=standard_floats),
                "bias": ParamConstraint(dtypes=standard_floats),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
            },
            default_devices=xla_devices,
        ),
    }


def _register():
    if not _XLA_AVAILABLE:
        registry.mark_unavailable("xla", _XLA_ERROR or "torch-xla not available")
        return
    registry.register(
        name="xla",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
