#!/usr/bin/env bash
# Symlink the Qwen-Image-2.1 ComfyUI repack into ComfyUI-XLA/models/.
#
# WHY SYMLINKS AND NOT extra_model_paths.yaml
# --------------------------------------------
# Both work, and this fork's ComfyUI is happy with either: folder_paths.py:415 walks
# with `os.walk(directory, followlinks=True)`, so symlinked FILES and symlinked
# DIRECTORIES are both followed, and extra_model_paths.yaml is loaded automatically by
# main.py:136-138 from utils/extra_config.py.
#
# The operator asked for the models/ layout, so that is what this builds: the tree
# under models/ is self-describing, works with any ComfyUI invocation including one
# that never reads extra_model_paths.yaml, and costs zero bytes.
#
# STORAGE (AGENTS.md §3). Every target is a symlink; no model byte is copied. The
# real bytes live in /dev/shm, a real 164 GiB tmpfs. Copying this set (~35 GiB) onto
# the root overlay would eat half of its ~68 GiB shared COW store — that is the
# failure mode this file exists to avoid.
#
# MISSING TARGETS ARE REPORTED, NOT FABRICATED. Three bf16 files were deliberately
# deleted from the staging tree to reclaim 36.6 GiB (see the staging README). This
# script does not download them and does not create dangling links that would make
# ComfyUI list a model it cannot open.
#
# REVERSIBLE BY: deleting the symlinks this script created. It touches no real bytes
# and no tracked file in the repo (models/ contents are not tracked; the 0-byte
# put_*_here placeholders are left alone).

set -uo pipefail

REPO=/kaggle/working/ComfyUI-XLA
REPACK=/dev/shm/comfyqwen          # official ComfyUI repack (real bytes)
HERETIC=/dev/shm/Qwen-Image-2.1   # heretic encoders + LoRAs (real bytes)

link() {  # link <target-subdir> <filename> <real-path>
  local sub="$1" name="$2" real="$3"
  local dest="$REPO/models/$sub/$name"
  if [ ! -e "$real" ]; then
    printf '  MISSING  %-14s %-58s (not on disk, not linked)\n' "$sub" "$name"
    return 1
  fi
  ln -sfn "$real" "$dest"
  printf '  linked   %-14s %-58s -> %s\n' "$sub" "$name" "${real#/dev/shm/}"
  return 0
}

echo "Qwen-Image-2.1 -> $REPO/models/  (symlinks only, zero bytes copied)"
echo
made=0; missing=0

# --- diffusion_models -------------------------------------------------------
link diffusion_models qwen_image_2.1_bf16.safetensors \
     "$REPACK/diffusion_models/qwen_image_2.1_bf16.safetensors" && made=$((made+1)) || missing=$((missing+1))
link diffusion_models qwen_image_2.1_int8_convrot.safetensors \
     "$REPACK/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))

# --- model_patches ----------------------------------------------------------
link model_patches qwen_image_2.1_fun_controlnet_union_bf16.safetensors \
     "$REPACK/model_patches/qwen_image_2.1_fun_controlnet_union_bf16.safetensors" && made=$((made+1)) || missing=$((missing+1))
link model_patches qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors \
     "$REPACK/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))

# --- text_encoders ----------------------------------------------------------
link text_encoders qwen3vl_8b_bf16.safetensors \
     "$REPACK/text_encoders/qwen3vl_8b_bf16.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3vl_8b_int8_convrot.safetensors \
     "$REPACK/text_encoders/qwen3vl_8b_int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3vl_8b_w4a8.safetensors \
     "$REPACK/text_encoders/qwen3vl_8b_w4a8.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors \
     "$REPACK/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors \
     "$REPACK/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))

# --- vae --------------------------------------------------------------------
link vae qwen_image_2.1_vae_bf16.safetensors \
     "$REPACK/vae/qwen_image_2.1_vae_bf16.safetensors" && made=$((made+1)) || missing=$((missing+1))

# --- heretic encoders (same models, abliterated weights) ---------------------
# Kept under distinct filenames alongside the vanilla ones so both are selectable.
# The int8 heretic 8B encoder is the one validated end-to-end through ComfyUI's
# CLIPType.QWEN_IMAGE path ("loaded completely; 8917.48 MB loaded, full load: True").
link text_encoders qwen3vl_8b_heretic_int8_convrot.safetensors \
     "$HERETIC/text_encoders/qwen3vl_8b_heretic_int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3vl_8b_heretic_w4a8.safetensors \
     "$HERETIC/text_encoders/qwen3vl_8b_heretic_w4a8.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3.5_9b_heretic_qwen_image_2.1_pe_i2i.int8_convrot.safetensors \
     "$HERETIC/text_encoders/qwen3.5_9b_heretic_qwen_image_2.1_pe_i2i.int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))
link text_encoders qwen3.5_9b_heretic_qwen_image_2.1_pe_t2i.int8_convrot.safetensors \
     "$HERETIC/text_encoders/qwen3.5_9b_heretic_qwen_image_2.1_pe_t2i.int8_convrot.safetensors" && made=$((made+1)) || missing=$((missing+1))

# --- loras ------------------------------------------------------------------
link loras RadianceChromeVoluptuous_QwenImage2.1_v1.0.safetensors \
     "$HERETIC/RadianceChromeVoluptuous_QwenImage2.1_v1.0.safetensors" && made=$((made+1)) || missing=$((missing+1))
link loras easternfemale_v2.1_v3_comfy.safetensors \
     "$HERETIC/../Qwen-Image-2.1-comfy/loras/easternfemale_v2.1_v3_comfy.safetensors" && made=$((made+1)) || missing=$((missing+1))

echo
echo "linked: $made    missing (left absent, no download): $missing"
