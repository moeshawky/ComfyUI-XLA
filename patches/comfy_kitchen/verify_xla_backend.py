"""Verify this fork's comfy-kitchen XLA backend is installed and dispatched.

Run directly, or via ../install_xla_backend.sh (which applies the edits and then
calls this). PJRT_DEVICE is forced to CPU here: this script must never claim
TPU chips, and backend dispatch is fully testable without them.

The dispatch assertions carry a negative control on purpose. Checking only that
an xla tensor selects the xla backend is not enough — a backend whose device
scoping is broken would also let cpu tensors reach it, silently stealing work
from the eager path that CUDA and CPU installs depend on. So cpu-must-stay-eager
is asserted as a real requirement, not an afterthought.
"""

from __future__ import annotations

import os
import sys

os.environ["PJRT_DEVICE"] = "CPU"
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v5litepod-8")


def main() -> int:
    import torch

    from comfy_kitchen.registry import registry

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
        if not ok:
            failures.append(label)

    print("== comfy-kitchen XLA backend verification ==")
    print("  registered:", sorted(registry.list_backends()))

    check("xla backend registered", registry.is_available("xla"), True)
    check("xla precedes eager in priority",
          registry._priority.index("xla") < registry._priority.index("eager"), True)

    # Dispatch. int8_linear's real parameter is `weight`, not `weight_q`; using
    # the wrong name makes the device constraint fall back to defaults and every
    # backend looks eligible, which hides the very bug this checks for.
    for device, want in (("xla", "xla"), ("cpu", "eager"), ("meta", "eager")):
        try:
            x = torch.randn(8, 64, device=device)
            w = torch.randint(-127, 127, (32, 64), dtype=torch.int8, device=device)
            ws = torch.rand(32, device=device)
            got = registry.get_capable_backend(
                "int8_linear", {"x": x, "weight": w, "weight_scale": ws}
            )
        except Exception as exc:  # a crash is a failure, not a fallback
            print(f"  FAIL  {device} tensor raised {type(exc).__name__}: {exc}")
            failures.append(f"{device} dispatch")
            continue
        check(f"{device} tensor dispatches to", got, want)

    # Negative control for the numerics: if the per-row activation scale is
    # dropped, error explodes by ~3 orders of magnitude. That separation is what
    # makes a small tolerance meaningful instead of decorative.
    try:
        from comfy_kitchen.backends.xla.quantization import int8_linear as xla_int8_linear

        torch.manual_seed(0)
        x = torch.randn(16, 64)
        w = torch.randint(-127, 127, (32, 64), dtype=torch.int8)
        ws = torch.rand(32) + 0.5

        out = xla_int8_linear(x, w, ws, out_dtype=torch.bfloat16)

        exact = (x.to(torch.float64) @ w.to(torch.float64).T) * ws.to(torch.float64)
        err = (out.to(torch.float64) - exact).abs().max().item()
        scale = exact.abs().max().item()
        rel = err / max(scale, 1e-9)
        print(f"  INFO  bf16 vs float64 reference: max_abs_err={err:.4g} rel={rel:.4g}")
        check("numerics within bf16 tolerance", rel < 0.05, True)

        check("output dtype honoured", out.dtype, torch.bfloat16)
        check("output shape", tuple(out.shape), (16, 32))
    except ImportError as exc:
        print(f"  FAIL  backend numerics not importable: {exc}")
        failures.append("numerics")

    print()
    if failures:
        print(f"FAILED: {failures}")
        print("If the backend files exist but dispatch is not xla, the registry")
        print("edits did not apply — a copied-in but unregistered backend goes")
        print("nowhere and still looks like a clean install.")
        return 1
    print("OK: xla backend installed, scoped, and numerically consistent.")
    print("TPU execution remains unproven; this ran on CPU only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())