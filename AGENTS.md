# AGENTS.md — ComfyUI-XLA on a Kaggle TPU v5e-8 (entry gate)

**Scope.** This file replaces the upstream ComfyUI `AGENTS.md` that arrived with the
`comfyanonymous/ComfyUI` master merge. Operator ruling: this is our fork; upstream's
guide (written for a desktop/web image app) does not govern this checkout.
Machine-level authority: `/kaggle/working/vllm-tpu/AGENTS.md` (canonical manual —
storage invariants, STEP 0–5, emergency protocol, custody contract); it wins conflicts.
Session read order: this file → `PROJECT-ROADMAP.md` §6 → the evidence files in §9.

## 1. The machine

- Kaggle v5e-8 slice: 8 chips, 2×4 topology (`TPU_CHIPS_PER_HOST_BOUNDS=2,4,1`),
  16 GiB HBM/chip, 384 GiB RAM, 224 vCPU. Not a dev box, not disposable. The
  2-hour watchdog counts REAL TPU compute (synchronised inference) only — attach, topology discovery, and weight loads do not count.
- A notebook kernel keeps this box alive: Cell 9 runs a blocking keepalive; state under `/tmp/moe-kaggle-tpu/`.
- Measured host stack (2026-10-07): torch 2.8.0+cpu, torch_xla 2.8.0, Python
  3.12.13, transformers 5.12.1, numpy 2.5.3. `README.md` documents only
  ROCm/XPU/CUDA install paths and has **no TPU section at all** — the merge took
  upstream's README and dropped the fork's XLA docs (ROADMAP Phase 5). Trust the
  measured versions above, never a README pin. Two stacks coexist — do not mix:
  this repo (system Python) and the isolated JAX-based vLLM venv at
  `/kaggle/temp/venvs/vllm-tpu`.

## 2. Never list

1. **Never touch `/tmp/moe-kaggle-tpu/`** (kernel state: `blocking-loop.json`,
   `heartbeat.json`, `kernel-bridge.json`). Never kill or restart the notebook kernel. Never
   hardcode its PID — read the `kernel_pid` field of `blocking-loop.json`.
2. **Never claim the TPU while another process owns it.** Check `lsof -t /dev/accel* /dev/vfio/*`
   or `tpu-audit` first; verify descriptors actually released before handing the TPU back.
3. **Never replace the protected stack** (torch, torch-xla, torchvision,
   torchaudio, jax, jaxlib, libtpu, tpu-info). All env/pip lifecycle goes through
   the `kaggle-backend` helper (`/usr/local/bin/kaggle-backend`): it enforces
   constraints and refuses direct orders for the protected set (no hand-rolled
   pip); the `vllm-tpu` venv is isolated (`include-system-site-packages = false`)
   and `pip check` must be rc 0 or 1 with exactly 4 audited overrides (jax/jaxlib/libtpu/numba).
4. **Never trust `df`/statvfs for capacity.** See §3.
5. **Never report a duration you did not measure.** Wall-clock only:
   `start=$(date +%s)` … `$(( $(date +%s) - start ))`; no number →
   "no wall-clock recorded". CPU ticks are not time.
6. **Never claim TPU correctness from device counts, topology discovery, or
   weight loads.** Minimum proof = a synchronised tensor op; for this repo,
   a generated image. See §6.
7. **Never add dependencies casually.** The merge already pulled in hard deps
   this host lacks (`comfy-aimdo==0.5.5`, `comfy-kitchen==0.2.37`,
   `comfyui-frontend-package==1.55.14`, alembic, SQLAlchemy, av, blake3,
   cryptography, pydantic-settings, PyOpenGL, comfy-angle), and the tree dies at
   import with `ModuleNotFoundError: No module named 'comfy_aimdo'`.
8. **Never delete `/tmp/libtpu_lockfile`** until you have confirmed no process
   owns the TPU (stale, but load-bearing).

## 3. Storage reality (`df` lies)

- `/`, `/tmp`, `/kaggle/temp` share ONE Docker overlay (dev 62). Real budget:
  a ~68 GiB COW store; the reported ~7.9 TiB is snapshot origin size, NOT free
  capacity. Exhaustion = session-wide catastrophe, not clean ENOSPC.
- `/dev/shm`: real tmpfs, ~164 GiB, RAM-backed, volatile.
- `/kaggle/working`: separate ext4, ~19.5 GiB — manifests, journals, small state.
- `/kaggle/input`: read-only NFS, the only durable large tier.
- XLA compile cache → `/dev/shm/xla_comfy_cache` (`COMFY_XLA_CACHE_PATH`);
  HF cache → `/dev/shm/hf` (`HF_HOME`) — both set by `xla_tpu_v5e8.sh`.
  **Never `/kaggle/temp`.** Never fill `/dev/shm` near its limit.
- The overlay takes small files only: scripts, logs, `/tmp/moe-kaggle-tpu/*`,
  `/kaggle/temp/emergency/*`.

## 4. TPU attach (measured on this host)

- `TPU_SKIP_MDS_QUERY=1` is mandatory for BOTH jax and torch_xla. Without it,
  `import torch_xla` (2.8.0) HTTP-GETs
  `metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env`,
  404s (the Cell-6 GCE shim serves only accelerator-type / instance-id /
  agent-worker-number), and dies with `OSError: Failed to get TPU metadata`.
- `TPU_ACCELERATOR_TYPE` must satisfy torch_xla's regex
  `^v(\d)([A-Za-z]?){7}-(\d+)$`. `v5litepod-8` → `tpu.version()==5` →
  1 logical core/chip → 8 chips = 8 XLA devices, v5 `LIBTPU_INIT_ARGS`
  applied. A malformed value raises at import.
- `comfy/model_management.py` imports torch_xla at MODULE scope: the env must be
  correct before `main.py` line 1 — that is why `xla_tpu_v5e8.sh` exists. Launch
  through it (full attach combo, §9); never hand-roll env for torch_xla.
- The 169.254 busybox metadata mock is banned on this host; the 127.0.0.1 GCE
  shim + its one `/etc/hosts` line is the only allowed hosts mutation.

## 5. Custody and experiments (TPU Flash Custody Contract)

- **Exactly one TPU owner at a time.** Source work may edit the tree and run
  CPU/static probes but may not claim chips; runtime work may launch/probe/measure
  but may not edit source. Only the coordinator/operator promotes to serving truth.
- **One experiment = one causal delta.** Never change source candidate and
  runtime baseline in the same experiment unless that pairing is itself the
  experiment. If provenance is wrong, the result is rejected, not interpreted.
- Diagnostics are on-demand: `tpu-audit` / `tpu-info` (one `--metric` per flag);
  nothing polls tpu-info periodically.
- **Guards fail closed; probes need negative controls.** Worked example from
  this repo: `xla_tpu_v5e8.sh` v1 parsed default `lsof` output with `awk 'NR>1'`
  (assumed header); a headerless PID line made it report "nobody owns the chips"
  and start anyway — failing OPEN on exactly the input it exists to catch. Fix:
  `lsof -t` (no header by definition) + a 5-case fake-lsof matrix, each row
  executed. A guard whose parsing rests on an unverified input-format assumption
  is a comment that sometimes raises, not a guard.
- If the primary stack cannot produce a synchronised op before the 2-hour
  boundary, use the machine's emergency path: Qwen 2.5 1.5B direct-greedy
  worker, no vLLM, no port 8000 — artifacts in `/kaggle/temp/emergency/`,
  notebook-resident rescue via `echo rescue|status|stop > /kaggle/temp/emergency/rescue.control`
  (refuses while `/dev/accel*` owned). Full protocol: the manual's EMERGENCY section.

## 6. Proof ladder (what counts as done)

1. Custody clear: chips unowned (`lsof -t`, `tpu-audit --metric duty_cycle_percent`).
2. Attach: `device_count()==8`, `tpu.version()==5`, `LIBTPU_INIT_ARGS` applied (manifest §3/F2 probe).
3. Compute: a real tensor op, synchronised (matmul + `.item()` round-trip).
   **Step 3 is the one that counts. Step 2 is not inference.**
4. Product: ComfyUI up, a real prompt produces an image (`xla.sync()` in
   `latent_preview.py` is what makes TPU progress visible).

## 7. Current port state (as of 2026-10-07 — re-measure, do not trust this list)

- The XLA surface is ABSENT from the tree: `grep -c -i xla` = 0 for
  `comfy/cli_args.py`, `comfy/model_management.py`, `latent_preview.py` —
  `CPUState.XLA`, `xla_mode()`, `get_xla_memory_info()`, `--xla`, `--xla_spmd`, and the `xla.sync()` callback all went out in the merge.
- All seven upstream hook points exist at known lines in
  `comfy/model_management.py`: `CPUState:53`, `get_torch_device:195`, `get_total_memory:316`,
  `get_free_memory:1811`, `should_use_fp16:1900`, `should_use_bf16:1967`, `supports_fp8_compute:2020`.
- The re-port checklist (4 files, 7 hunks, plus the two grep false positives to
  NOT re-add — `flux_xlabs_mistoline` controlnet, the `XLabs-AI/x-flux` URL) is
  `docs/XLA_PORT_MANIFEST.md` §2/§4.
- Phase 1 fixed five defects here, each with an executable probe — don't
  regress them: F1 `args.xla_eager` read with no argparse producer (crashed
  every platform), F2 MDS 404 at import, F3 v3-era SPMD divisor 3.0 hardcoded,
  F4 compile cache pinned to `/tmp`, F5 SPMD mesh flattened 2×4 to (N,1).
  Re-run the manifest probes after any change.
- `comfy/ldm` went 11 → 52 entries in the merge (upstream master: 58 at manifest
  time). Reaching modern models is ROADMAP Phase 4 — a rebase, one causal delta, not mixed with other work.

## 8. Code rules that replaced upstream's

- A new abstraction must justify itself. **"Untestable because claiming a device
  is forbidden" is a legitimate justification** under the custody contract — state
  it where the helper lives. (Replaces upstream's blanket "unnecessary helper layers will be rejected".)
- Use the existing helpers — `comfy.quant_ops`, `comfy.model_management`, Comfy
  Kitchen / `comfy-kitchen`, `comfy_aimdo` — before writing parallel code; they
  are mandatory for the work this fork actually does. (Replaces "prefer fewer dependencies".)
- Backend matrix here: **XLA/TPU v5e-8 is the primary target and the reason
  this fork exists**; CPU is the probe path. CUDA/ROCm/MPS/DirectML/XPU are
  not this fork's concern. (Replaces upstream's multi-backend checklist.)
- Dropped as upstream process, not machine constraints: the "code must look
  hand-written … AI-generated code will be rejected automatically" review rule; node/UX/workflow rules aimed at the desktop/web app.
- Kept (true here too): small diffs, narrowest code path, no dead branches,
  backward-compatible node changes, module-scope imports — except where a
  backend probe genuinely needs lazy import (the notebook's rescue thread imports
  torch/torch_xla/transformers only inside its worker thread, only when the control file appears).

## 9. Where the evidence lives

- Machine manual (canonical): `/kaggle/working/vllm-tpu/AGENTS.md` — storage
  invariants, STEP 0–5, emergency protocol, custody contract.
- Bootstrap record: `/kaggle/working/vllm-tpu/notebook/stage0-9.txt` (audit,
  secrets, ssh/tunnel, toolchain, `kaggle-backend` helper, vLLM env, handoff,
  stage-9 supervisor). Stages 1–2 read Kaggle user secrets (`ssh_salt`,
  `cloudflared_tunnel_salt`) — values never serialised. Credential-bearing files
  in the repo (forge tokens in `frpc.toml`/docs) are intentional: private repo, never push public.
- Port evidence: `docs/XLA_PORT_MANIFEST.md` (hunks, F1–F5, executed probes, and
  what is explicitly NOT proven — static/CPU evidence only).
- Plan and scope: `PROJECT-ROADMAP.md` (phases, scope-creep defences, drift
  checkpoints — re-read before touching a file).
- Launcher: `xla_tpu_v5e8.sh` (env combo, MDS failure mode, custody guard,
  cache/HF tiering).
- Live runtime state (read, never write): `/tmp/moe-kaggle-tpu/blocking-loop.json`
  (kernel PID), `heartbeat.json`, `telemetry.log`; emergency artifacts in `/kaggle/temp/emergency/`.

## 10. Session checklist

1. Read this file + `PROJECT-ROADMAP.md` §6.
2. Check custody: `lsof -t /dev/accel* /dev/vfio/*`,
   `tpu-audit --metric duty_cycle_percent`.
3. Launch via `xla_tpu_v5e8.sh` (env must be set before any Python that imports
   torch_xla). No one-liners.
4. Run the §6 proof ladder for the phase you claim.
5. Record wall-clock for every phase; "no wall-clock recorded" beats a guess.

**Last words (these survive context rot):** `df` lies. The TPU has exactly one
owner. Attach is not inference. The kernel must survive you. Measure it or say
you didn't.
