#!/usr/bin/env bash
# run_x86_original_imtalker.sh -- PersonaPlex + OFFICIAL IMTalker real-time avatar on an
# x86_64 NVIDIA GPU (RunPod / Lambda / a local workstation).
#
# Same 4-bit PersonaPlex front end as run_x86.sh (int4 trunk, Robert_5 one-pass depformer,
# GAN Mimi decoder, optional TensorRT SEANet, Mimi CUDA graph), but the avatar half is the
# ORIGINAL IMTalker stack instead of the causal motion student + distilled renderer:
#
#   PersonaPlex reply hidden -> Helium->wav2vec frontend adapter
#     -> FM generator (official cbsjtu01/IMTalker generator.ckpt)
#     -> FLAME renderer (official cbsjtu01/IMTalker renderer.ckpt)
#
# prepare_x86.sh downloads the extra weights (~5.6 GB):
#   weights/generator_original.ckpt   weights/renderer_original.ckpt
#   weights/wav2vec2-base-960h/       weights/adapter_last.pt  (+ .adapter.env)
#
#   ./run_x86_original_imtalker.sh                      defaults (port 8991)
#   PORT=8080 ./run_x86_original_imtalker.sh            another port (8998 is refused)
#   NFE=10 A_CFG=3.0 ./run_x86_original_imtalker.sh     official-app sampling (slower)
#   RENDERER_PRECISION=bf16 ./run_x86_original_imtalker.sh   less VRAM for the renderer
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"; W="$HERE/weights"
[ -s "$HERE/.snap_path" ] && [ -s "$HERE/.python_path" ] || { echo "run ./prepare_x86.sh <hf_token> first"; exit 1; }
PY="${PY:-$(cat "$HERE/.python_path")}"   # the Python prepare_x86.sh checked
SNAP="$(cat "$HERE/.snap_path")"
PORT="${PORT:-8991}"   # 8998 is refused by the server (reserved for a legacy production server)
export HF_HOME="${HF_HOME:-$HERE/hf}"

for f in generator_original.ckpt renderer_original.ckpt adapter_last.pt; do
  [ -s "$W/$f" ] || { echo "missing weights/$f -- run ./prepare_x86.sh <hf_token> first"; exit 1; }
done
[ -d "$W/wav2vec2-base-960h" ] || { echo "missing weights/wav2vec2-base-960h -- run ./prepare_x86.sh first"; exit 1; }

# Adapter architecture, detected and strict-load-validated by prepare_x86.sh.
# Defaults match the production UniTalk 12-layer adapter if .adapter.env is absent.
ADAPTER_TYPE=unitalk_last_layer; ADAPTER_NUM_LAYERS=12; ADAPTER_DROPOUT=0.0
[ -s "$HERE/.adapter.env" ] && . "$HERE/.adapter.env"

# RunPod (and other) images export HF_HUB_ENABLE_HF_TRANSFER=1; huggingface_hub then refuses to
# download anything at all unless the hf_transfer package is importable. setup_env_x86.sh installs
# it (requirements-x86-extra.txt), but if this Python does not have it, turn the accelerator off
# instead of failing the download.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" != "0" ] && ! "$PY" -c 'import hf_transfer' 2>/dev/null; then
  echo "    note: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed -> disabling it"
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
# compiled-kernel caches next to the package so they survive on the persistent volume
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export DESIGN_A_DIR="$SRC/niloys" IMTALKER_DIR="$SRC/imtalker"
# NOTE: src/pylibs is deliberately NOT on the path -- the sphn there is an aarch64 build.
export PYTHONPATH="$SRC/niloys:$SRC/moshi_pkg:${PYTHONPATH:-}"
unset TRY19_DEC_CKPT
export NATIVE_PASSTHROUGH=1

# Keep the native IMTalker path inside imtalker_native_passthrough: FM generator, full
# FLAME renderer and the frontend adapter all stay loaded (the student build sets these
# to 1 to skip them). --avatar_backend original does the same inside the wrapper; the
# exports here make the intent explicit and cover older clones.
export TRY19_NO_FM=0 TRY19_NO_FULL_RENDERER=0 TRY19_NO_FRONTEND=0
export TRY19_PD="$W/pd_quant_onpol_robert5_plaus2.pt"
export TRY19_MIMI_DEC_FT="$W/mimi_dec_gan_robert5.pt"
export TRY19_MIMI_GRAPH="${TRY19_MIMI_GRAPH:-1}" TRY19_MIMI_PARTS="${TRY19_MIMI_PARTS:-seanet_quant}" TRY19_MIMI_CAPTURE_AT_INIT="${TRY19_MIMI_CAPTURE_AT_INIT:-1}"
export TRY19_CODE_GUARD="${TRY19_CODE_GUARD:-1}"

# int4 trunk: ATen's tinygemm kernels are not built for every CUDA architecture, so
# prepare_x86.sh probed them and wrote the verdict. An explicit value in the environment wins.
_int4_user="${TRY19_INT4_G:-}"; _gate_user="${TRY19_DEP_GATING_INT4:-}"
if [ -s "$HERE/.gpu_probe.env" ]; then set -a; . "$HERE/.gpu_probe.env"; set +a; fi
export TRY19_INT4_G="${_int4_user:-${TRY19_INT4_G:-32}}"
export TRY19_DEP_GATING_INT4="${_gate_user:-${TRY19_DEP_GATING_INT4:-1}}"

# TensorRT SEANet decoder: optional here. Without it Mimi's PyTorch SEANet is used.
if [ -s "$W/seanet_w12_fp32.engine" ]; then
  export TRY19_TRT_SEANET="$W/seanet_w12_fp32.engine"
else
  unset TRY19_TRT_SEANET
  echo "note: no weights/seanet_w12_fp32.engine -- using Mimi's PyTorch SEANet decoder"
fi

# Eye-blink compositing (motion-map blink loop decoded by the official renderer).
BLINK=()
if [ "${ENABLE_BLINK:-1}" = "1" ] && [ -s "$W/blink_motion.pt" ]; then
  BLINK=(--enable_eye_blink_composite --blink_motion_path "$W/blink_motion.pt")
fi
# torch.compile the fused renderer (long startup compile, faster steady state).
COMPILE=()
[ "${COMPILE_RENDERER:-0}" = "1" ] && COMPILE=(--compile_renderer)

mkdir -p "$SRC/niloys/live/static" "$HERE/captures"
ln -sfn "$SRC/imtalker/static/assets" "$SRC/niloys/live/static/assets"
ln -sfn "$SRC/niloys/static/index_try21_no_idle_dissolve.html" "$SRC/niloys/live/static/index.html"
echo "x86 original-IMTalker: trunk int4 G=$TRY19_INT4_G + robert5 one-pass depformer + GAN decoder$([ -n "${TRY19_TRT_SEANET:-}" ] && echo ' (TensorRT)') | adapter=$ADAPTER_TYPE/$ADAPTER_NUM_LAYERS nfe=${NFE:-5} cfg=${A_CFG:-1.34} voice=${VOICE:-Robert_5.pt} port=$PORT"
exec "$PY" -u "$SRC/niloys/live/imtalker_try19_student_live.py" \
  --live_module imtalker_native_passthrough \
  --avatar_backend original --dep_q 8 \
  --host 0.0.0.0 --port "$PORT" \
  --html_path "$SRC/niloys/static/index_try21_no_idle_dissolve.html" \
  --generator_path "$W/generator_original.ckpt" --renderer_path "$W/renderer_original.ckpt" \
  --adapter_path "$W/adapter_last.pt" \
  --adapter_type "$ADAPTER_TYPE" --adapter_num_layers "$ADAPTER_NUM_LAYERS" --adapter_dropout "$ADAPTER_DROPOUT" \
  --adapter_window_mode "${ADAPTER_WINDOW_MODE:-tail}" --adapter_future_steps "${ADAPTER_FUTURE_STEPS:-6}" \
  --ref_path "$W/ref.jpeg" --wav2vec_model_path "$W/wav2vec2-base-960h" \
  --moshi_root "$W" --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$SNAP/model.safetensors" --mimi_weight "$W/mimi.safetensors" \
  --tokenizer "$W/tokenizer_spm_32k_3.model" \
  --text_prompt "${TEXT_PROMPT:-You are Robert, an assistant from RB Labs. You have knowledge about crypto, finance, investment, technology, and general topics. Answer every question completely, descriptively, and with useful detail.}" \
  --voice_prompt "${VOICE:-Robert_5.pt}" --voice_prompt_dir "$W/voices" \
  --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 0 \
  --audio_chunk_sec 0.96 --wav2vec_sec 0.96 --fm_chunk_frames 24 --helium_deque_size 100 \
  --prebuffer_chunks "${PREBUFFER:-3}" --render_sub_batch "${RENDER_SUB_BATCH:-8}" \
  --renderer_precision "${RENDERER_PRECISION:-fp32}" \
  --frame_q_backpressure "${FRAME_Q:-3}" --buffer_ms 80 --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 \
  --motion_ref_blend "${MOTION_REF_BLEND:-0.0}" \
  --a_cfg_scale "${A_CFG:-1.34}" --nfe "${NFE:-5}" --seed 42 \
  --fp32 --tf32 \
  --silence_helium_path "$W/silence_helium_mean.pt" \
  --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
  --capture_dir "$HERE/captures" \
  ${BLINK[@]+"${BLINK[@]}"} ${COMPILE[@]+"${COMPILE[@]}"}
