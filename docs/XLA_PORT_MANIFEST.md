# XLA Port Manifest — ComfyUI-XLA on TPU v5e-8

*What this fork actually patches, where, and what a rebase onto current upstream
ComfyUI has to carry. Every entry below was located by reading the source on this
checkout (`origin/radna0/ComfyUI-XLA`, HEAD `ec8a77a`, upstream base merged
2024-12-15). Nothing here is quoted from upstream docs.*

---

## 0. Why this document exists

`grep -c -i xla` on **current upstream ComfyUI** returns:

| Upstream file | xla hits |
|---|---|
| `comfy/cli_args.py` | **0** |
| `comfy/model_management.py` | **0** |

Upstream ComfyUI has no XLA/TPU support at all. This fork is not a convenience
wrapper — it is the only thing standing between ComfyUI and a TPU. So "update
ComfyUI" and "keep TPU support" are the same problem, and this file is the
checklist for solving it without losing the TPU side.

---

## 1. Model-zoo gap (measured, both directions)

| | `comfy/ldm` entries |
|---|---|
| **This fork** | 11 — `audio`, `aura`, `cascade`, `common_dit.py`, `flux`, `genmo`, `hydit`, `lightricks`, `models`, `modules`, `util.py` |
| **Upstream master** (GitHub contents API) | **58** — adds `ace`, `anima`, `boogu`, `chroma`, `chroma_radiance`, `cogvideo`, `cosmos`, `depth_anything_3`, `ernie`, `hidream`, `hidream_o1`, `hunyuan3d`, `hunyuan3dv2_1`, `hunyuan_video`, `ideogram4`, `joyimage`, `kandinsky5`, `krea2`, `lens`, `lumina`, `mage_flow`, `minimax`, `minimax_music`, `mmaudio`, `moge`, `omnigen`, `pixart`, `pixeldit`, `qwen_image`, `qwen_image21`, `rt_detr`, `sam3`, `sam3d_body`, `seedvr`, `sensenova`, `supir`, `trellis2`, `triposplat`, `wan`, `yue2`, … |

**Consequence:** "modern models" and "TPU support" are in direct tension in this
checkout. The fork can reach the TPU; it cannot name most of the architectures
that matter. Porting the XLA surface forward is what resolves that tension, and
it is a rebase, not a patch. This file exists to make that rebase mechanical.

---

## 2. Real XLA surface — 4 files, 7 functional hunks

`grep -rn -i xla --include=*.py` over the checkout reports 5 files. **Two of
those five are false positives**; read the actual lines before trusting a grep:

| File | Verdict |
|---|---|
| `comfy/cli_args.py` | **REAL** — `xla_group` + the flags added here |
| `comfy/model_management.py` | **REAL** — the bulk of the port |
| `latent_preview.py` | **REAL** — `prepare_callback` |
| `comfy/controlnet.py` | **FALSE POSITIVE** — matched `xla` inside `load_controlnet_flux_xlabs_mistoline`. No XLA runtime. |
| `comfy/ldm/flux/controlnet.py` | **FALSE POSITIVE** — matched `xla` inside the URL `github.com/XLabs-AI/x-flux`. No XLA runtime. |

### H1 — `comfy/cli_args.py`: backend selection flags
```
--xla              # XLA for everything
--xla_spmd         # XLA + SPMD/FSDPv2   (mutually exclusive with --xla)
--xla_eager        # eager mode           (added 2026-10-07, see §3)
--xla_eager_compile# alias for the above
--xla_cache_path   # compile cache dir    (added 2026-10-07)
--xla_spmd_mem_divisor # SPMD HBM divisor (added 2026-10-07)
--xla_mesh         # SPMD mesh shape      (added 2026-10-07)
```
`--xla` and `--xla_spmd` live in an `add_mutually_exclusive_group`. The eager
flags deliberately live **outside** it — they are modifiers of a backend, not
competing backends, so `--xla --xla_eager` must stay expressible.

### H2 — `comfy/model_management.py`: device selection
- `class CPUState` gains `XLA = 3` (alongside GPU/CPU/MPS).
- The module-scope init block: `import torch_xla as xla`,
  `import torch_xla.core.xla_model as xm`, `from torch_xla import runtime as xr`,
  then `xr.initialize_cache(<path>)`, then `cpu_state = CPUState.XLA`.
- `get_torch_device()` returns `xla.device()` for `CPUState.XLA`.
- **This import is module-scope.** Every XLA environment variable must be correct
  before `main.py` line 1. That is why `xla_tpu_v5e8.sh` exists.

### H3 — `comfy/model_management.py`: memory accounting
- `get_xla_memory_info(dev)` — non-SPMD branch calls `xm.get_memory_info(dev)`
  (returns `bytes_used` / `bytes_limit`); SPMD branch sums `tpu_info` chip usage.
- `get_total_memory()` and `get_free_memory()` both branch on `CPUState.XLA`.
- `vram_state = VRAMState.DISABLED` when `cpu_state` is neither GPU nor XLA — XLA
  gets the full VRAM-state machinery, unlike MPS/CPU.

### H4 — `comfy/model_management.py`: precision / capability predicates
- `xla_mode()` helper.
- `should_use_fp16` / `should_use_bf16`: XLA returns **False** for fp16,
  **True** for bf16 — i.e. bf16 is the intended XLA compute precision.
- `supports_fp8_compute`: returns **True** unconditionally under XLA.

### H5 — `comfy/model_management.py`: SPMD mesh
```python
num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(np.arange(num_devices), (num_devices, 1), ("fsdp", "model"))
xs.set_global_mesh(mesh)
```
The `'fsdp'` axis name is **mandatory** — XLA shards weights and activations
along it.

### H6 — `latent_preview.py`: `prepare_callback`
```python
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    _prepare_spmd_partition_spec,
    SpmdFullyShardedDataParallel as FSDPv2,   # note the alias
)
if args.xla_spmd and not isinstance(model.model.diffusion_model, FSDPv2):
    model.model.diffusion_model = FSDPv2(model.model.diffusion_model)
...
if args.xla or args.xla_spmd:
    xla.sync()          # blocks until queued TPU work has executed
```
Two traps for anyone re-porting this:
1. The class is named `SpmdFullyShardedDataParallel`; `FSDPv2` is a local alias.
   `grep 'class FSDPv2'` finds nothing and looks like a missing dependency. It is not.
2. `xla.sync()` is defined in `torch_xla/torch_xla.py:78` and re-exported by
   `from .torch_xla import *`. It is the only thing making progress visible.

---

## 3. Defects found and fixed on this host (2026-10-07)

Each is one causal delta, each verified by a probe that can fail.

### F1 — `AttributeError: 'Namespace' object has no attribute 'xla_eager'`
**Severity: total.** `model_management.py` consumed `args.xla_eager` /
`args.xla_eager_compile` at module scope with **no argparse producer**, and the
read sat *outside* the `if args.xla or args.xla_spmd:` guard — so the crash hit
**every platform**, CPU and CUDA included, not just TPU.

*Before:* `python3 -c "import comfy.model_management"` → `PRODUCER_EXIT:1`,
stderr ends `AttributeError: 'Namespace' object has no attribute 'xla_eager'`.

*Fix:* declare both flags in `cli_args.py`, and read them via
`getattr(args, "xla_eager", False)` so a missing optional flag degrades to
"disabled" instead of killing the process.

*After:* `PROBE2_EXIT:0`, `cpu_state=CPUState.CPU`, `total_vram_MiB = 386904`.

### F2 — `OSError: Failed to get TPU metadata` on `import torch_xla`
**Severity: total on this host.** torch_xla 2.8's `_setup_default_env()` calls
`tpu.version()` → `tpu.get_tpu_env()`, which HTTP-GETs
`http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env`
unless `TPU_SKIP_MDS_QUERY` is set. This host's GCE shim does not serve that
path → 404 → `EnvironmentError`.

*Before:* `import comfy.model_management` with `--xla` →
`requests.exceptions.HTTPError: 404 ... /computeMetadata/v1/instance/attributes/tpu-env`.

*After (same env, `TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v5litepod-8`):*
```
PROBE7_EXIT:0
  tpu.version()              = 5
  num_logical_cores_per_chip = 1
  num_available_chips (PCI)  = 8
  num_available_devices      = 8
  LIBTPU_INIT_ARGS           = --xla_tpu_use_enhanced_launch_barrier=false
    --xla_latency_hiding_scheduler_rerun=1
    --xla_tpu_prefer_async_allgather_to_allreduce=true
    --xla_tpu_enable_flash_attention=false
    --xla_enable_async_all_gather=true
    --xla_enable_async_collective_permute=true
```
`num_available_chips() == 8` is read from PCI sysfs and means the chips are
physically present. It is **not** proof of inference — see §5.

*Note on the regex:* `tpu.version()` matches `^v(\d)([A-Za-z]?){7}-(\d+)$`
against `TPU_ACCELERATOR_TYPE`. `v5litepod-8` matches and yields `5`. A malformed
value raises and the whole import dies.

### F3 — the SPMD memory divisor was a hardcoded v3 constant
The original comment read *"Tested on TPU v3-8, given 8 cores, only 3/8 of the
memory is available in SPMD mode"* followed by a bare `mem_reserved /= 3;
mem_total /= 3`. That number drives **every** ComfyUI model-placement decision.
It is an empirical measurement from a different chip generation, not a law of
SPMD, and it was silently applied to every TPU including v5e.

*Fix:* exposed as `--xla_spmd_mem_divisor` (default `3.0`, byte-identical to
before), validated, and logged with raw vs usable totals so a v5e calibration is
observable rather than guessed. Nothing was re-tuned without a measurement.

*Probe:* `divisor=3.0 → 3.0`, `1.0 → 1.0`, `0 → 3.0`, `-2 → 3.0`,
`'abc' → 3.0`, `<attr absent> → 3.0`, each with a warning.

### F4 — the compile cache was pinned to `/tmp`
`xr.initialize_cache("/tmp")` with no override. On this host `/`, `/tmp` and
`/kaggle/temp` are one Docker overlay with a scarce COW store, and its reported
free space is fiction; `/dev/shm` is a real 164 GiB tmpfs.

*Fix:* precedence `XLA_COMFY_CACHE_PATH` env > `--xla_cache_path` > `/tmp`
(historical default preserved). `initialize_cache` only sets
`XLA_PERSISTENT_CACHE_PATH` / `XLA_PERSISTENT_CACHE_READ_ONLY`, so the directory
is created here.

*Probe (consumer-vantage read, fresh `stat` of the created directory):*
```
A default, no env/flag  -> LOG XLA compilation cache: /tmp
B --xla_cache_path=/dev/shm/xla_comfy_cache
                        -> LOG XLA compilation cache: /dev/shm/xla_comfy_cache
C env=/dev/shm/env_wins + flag=/dev/shm/flag_loses
                        -> LOG XLA compilation cache: /dev/shm/env_wins

  EXISTS  /tmp
  EXISTS  /dev/shm/xla_comfy_cache
  EXISTS  /dev/shm/env_wins
  ABSENT  /dev/shm/flag_loses      <- negative control: env really beat the flag
```

### F5 — the SPMD mesh discarded the 2-D host topology
The mesh was always `(num_devices, 1)`. A v5e-8 Kaggle slice is **2x4**
(`TPU_CHIPS_PER_HOST_BOUNDS=2,4,1`), so the flat mesh throws away the real
interconnect topology.

*Fix:* `--xla_mesh 2,4`. Resolved by the pure function `_xla_mesh_shape()`, which
puts the mandatory `'fsdp'` axis first and appends the rest.

*Probe:* `None@8 → ((8,1),('fsdp','model'))`, `'2,4'@8 → ((2,4),('fsdp','mesh0','mesh1'))`,
`'2,4'@4 → ((4,1),...)` + warning, `'garbage'@8 → ((8,1),...)` + warning,
`'0,8'@8 → ((8,1),...)` + warning.

---

## 3a. A defect this manifest's own author shipped and caught

Recorded because the evidence trail is the point.

`xla_tpu_v5e8.sh` v1 parsed default `lsof` output with `awk 'NR>1 {...}'` to skip
the header row. A negative control — a fake `lsof` emitting a single PID line and
no header — showed the guard reporting **"nobody owns the chips"** and starting
anyway. That is a guard that **fails open on exactly the input it exists to
catch**, and it came from a copy-paste habit, not from reading lsof's contract.

Fixed by switching to `lsof -t` (terse, field-selected: PIDs only, one per line,
no header by definition), so no header assumption exists to be wrong.

Guard matrix after the fix, each row executed:

| Fake `lsof` behaviour | Expected | Got |
|---|---|---|
| header present, 1 holder | refuse, exit 1 | PASS |
| **no header, 1 holder** | refuse, exit 1 | **PASS (was FAIL)** |
| no header, 2 holders | refuse, exit 1 | PASS |
| nothing, exit 1 | proceed, exit 0 | PASS |
| nothing, exit 0 | proceed, exit 0 | PASS |

The lesson generalises past this script: **a guard whose parsing depends on an
unverified assumption about its input's formatting is not a guard.** It is a
comment that sometimes raises.

---

1. `comfy/cli_args.py` — re-add `xla_group` and all seven flags. Keep
   `--xla`/`--xla_spmd` mutually exclusive; keep eager/cache/mesh/divisor outside.
2. `comfy/model_management.py` — re-add `CPUState.XLA`, the module-scope torch_xla
   import, `get_torch_device`, `get_xla_memory_info`, and the `XLA` branches in
   `get_total_memory`, `get_free_memory`, `should_use_fp16`, `should_use_bf16`,
   `supports_fp8_compute`, `xla_mode()`, and the `vram_state` rule.
3. `latent_preview.py` — re-add `prepare_callback`'s FSDPv2 wrap and `xla.sync()`.
4. Do **not** re-add the two false positives in §2.
5. Re-run the §3 probes. They are cheap and each one can fail.

Upstream `model_management.py` has been refactored heavily since 2024-12 (device
abstraction, `torch.compile` paths). Expect conflicts in `get_free_memory` and
`get_torch_device` specifically — those are the two the platform kept rewriting.

---

## 5. What is NOT proven here

Everything above is **static/CPU evidence**. Per the TPU Flash Custody Contract,
source edits and live-TPU correctness belong to different roles, and no TPU was
claimed while producing this document. Specifically unproven:

- `xm.get_memory_info(dev)` on the real TPU backend. On a CPU PJRT backend it
  raises `RuntimeError: Bad StatusOr access: UNIMPLEMENTED: GetAllocatorStats is
  not supported`; whether the v5e TPU backend supports allocator stats is
  unverified here.
- Whether SPMD `tpu_info` totals, at divisor 3.0 or 1.0, let a real model load on
  v5e-8.
- Whether any XLA graph compiles at all on v5e, and at what wall-clock cost.
- That 8 chips become 8 XLA devices at runtime — `num_available_devices() == 8`
  counts PCI-attached chips; it does not prove a client initialised.

The runtime agent's probes, in order:

```bash
# 1. custody: chips must be unowned
tpu-audit --metric duty_cycle_percent

# 2. attach: exactly one process owns the chips, 8 of them
source ./xla_tpu_v5e8.sh   # or export the env it prints, then:
python3 -c "import torch_xla as xla; print(xla.device_count())"

# 3. real compute: an actual tensor op, synchronised
python3 -c "
import torch, torch_xla as xla
d = xla.device(); x = torch.ones(1024,1024, device=d)
print('matmul ok:', bool((x@x).sum().item() > 0))"

# 4. first image: ComfyUI up, then a real prompt through the UI/API
./xla_tpu_v5e8.sh --listen 0.0.0.0 --port 8188
```

Step 3 is the one that counts. Attach is not inference.