#!/bin/bash
# LTX25-DIT-ATTN-FLASH (#1612) -- the PIXEL A/B, and the speed A/B it also settles.
#
# `scripts/ltx25-dit-attn-flash-ab.sh` is this file's sibling and answers a
# different question. It caps each arm at 13 forwards, which is enough for a
# per-forward median and produces NO FRAMES. #1549 shipped on that: a kernel
# swap that is explicitly not bit-identical on CUDA, gated only by a
# reduced-dimension host-vs-device parity case, with nothing at all said about
# what the model RENDERS. A diffusion model has no token gate, so there is no
# discrete output to fall back on. This file renders both arms to completion and
# compares the pixels.
#
# THREE RENDERS, and the third is the point.
#
#   1. flash      VLLM_LTX2_DIT_FLASH_ATTN=1   the arm that ships today
#   2. naive      VLLM_LTX2_DIT_FLASH_ATTN=0   the arm #1549 replaced
#   3. flash-ctl  VLLM_LTX2_DIT_FLASH_ATTN=1   flash AGAIN, same binary, same seed
#
# Without (3) an arm-to-arm difference cannot be attributed to the kernel. Two
# runs of one configuration measure what the BOX does on its own -- cuBLAS split
# reductions, allocator-dependent kernel selection, anything nondeterministic
# anywhere in a 120-forward denoise plus a VAE decode. If (3) is bit-identical
# to (1) the noise floor is exactly zero and every bit of the flash-vs-naive
# delta is the swapped op. If (3) differs from (1) by as much as (2) does, the
# swap changed nothing the machine does not change by itself. Either reading is
# an answer; neither is available from two renders.
#
# ORDER: flash, naive, flash-ctl. The naive arm is ~6x the wall clock of a flash
# arm and it is the one whose loss leaves no A/B at all, so it is taken second,
# while the box is known good, rather than last. The control is last because it
# is the only one recoverable in a short follow-up lease: the build cache below
# is keyed on the source sha, so a resumed run reaches a render in minutes.
#
# The 20260821T092516Z attempt lost the worker at forward 20 with no memory
# trace and no guard, so it could not say afterwards what had happened. This one
# writes a MemAvailable trace per arm and stops an arm that crosses a floor.
# It does NOT run under `runguard.py --stack-period`: that sampler ptrace-stops
# every thread and cost the recorded 47.84 s denominator ~3.2% (spec section
# 7.1). Both arms here are instrumented identically and neither is sampled, so
# the ratio needs no correction.
set -u
T0=$(date +%s)
say() { echo "[pixab +$(( $(date +%s) - T0 ))s] $*"; }
W=/workspace/ltx25-attnflash
FULL=/workspace/ltx25-fullmodel        # checkpoints and the text-encoder config
SRC=/root/src-pixab
BLD=/root/build-pixab
CK=/root/ckpt
OUT=$W/pixel-ab/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT" "$CK"
export DEBIAN_FRONTEND=noninteractive
say "OUT=$OUT"
{
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "started=$(date -Is)"
} > "$OUT/PROVENANCE"

# A LIVENESS LINE, and nothing more. The build redirects to a file and a single
# 42 GB checkpoint copy takes minutes, so this job can produce no stdout at all
# for the better part of an hour, and a lease with an idle timeout would kill a
# healthy run. It prints ONLY the elapsed time: it reports no count and no
# progress, because a line emitted on a fixed cadence cannot distinguish work
# from a hang and must not be read as if it could. The render loop below prints
# the forward count separately, and that one CAN stop advancing.
( while :; do sleep 120; echo "[pixab-alive +$(( $(date +%s) - T0 ))s]"; done ) &
HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

say "=== [0] the box ==="
uname -m; nproc; free -g | head -2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
df -h / /root /workspace 2>&1 | head -6

say "=== [1] tools ==="
# ffmpeg matters beyond the mp4: ltx2-gen exits 127 from its absence AFTER every
# frame and the wav are on disk (#1149), so its absence reads as a failed render
# even though the artifacts this file compares are already written. numpy is
# what scripts/ltx25-render-compare.py needs and it is the only dependency it
# has; the PPM and WAV readers there are written out precisely so that no image
# or audio library has to be present in a leased worker.
apt-get update -qq >/root/apt.log 2>&1
apt-get install -y -qq ffmpeg python3-numpy >>/root/apt.log 2>&1
echo "apt_rc=$?"
for t in ffmpeg ffprobe python3; do printf '%s=' "$t"; command -v "$t" || echo MISSING; done
python3 -c 'import numpy;print("numpy",numpy.__version__)' 2>&1 | tail -1

say "=== [A] CUDA toolkit, resolved BEFORE anything probes it ==="
CUDATK=/root/cudatk
STAGED=/workspace/a3/cuda-staged
need_ok() { [ -x "$1/bin/nvcc" ] && [ -f "$1/targets/sbsa-linux/lib/libcublasLt.so" ]; }
TKLIB=""
for r in /usr/local/cuda /usr/local/cuda-13.0 "$CUDATK"; do
  need_ok "$r" && TKLIB="$r" && say "complete toolkit: $r" && break
done
if [ -z "$TKLIB" ]; then
  for r in /usr/local/cuda /usr/local/cuda-13.0; do
    [ -x "$r/bin/nvcc" ] || continue
    LD="$r/targets/sbsa-linux/lib"
    [ -d "$LD" ] && [ -w "$LD" ] || continue
    for lib in libcublasLt libcublas; do
      f=$(ls "$STAGED"/targets/sbsa-linux/lib/$lib.so.13* 2>/dev/null | grep -v static | head -1)
      [ -n "$f" ] || continue
      say "repairing $r: adding $(basename "$f")"
      cp -f "$f" "$LD"/ && ( cd "$LD" && b=$(basename "$f"); ln -sf "$b" "$lib.so.13"; ln -sf "$b" "$lib.so" )
    done
    need_ok "$r" && TKLIB="$r" && break
  done
fi
if [ -z "$TKLIB" ] && [ -d "$STAGED/bin" ]; then
  if ! need_ok "$CUDATK"; then
    say "staging a COMPLETE toolkit locally (~4.9G)"
    mkdir -p "$CUDATK"; cp -a "$STAGED"/. "$CUDATK"/ 2>/dev/null
    # The share serves file_mode=0664: every binary copied off it arrives
    # NON-EXECUTABLE, and an nvcc that cannot run reads as an nvcc that is absent.
    find "$CUDATK/bin" "$CUDATK/nvvm/bin" -type f -exec chmod 0755 {} + 2>/dev/null
    # CIFS stores no symlinks; find_package wants libX.so and libX.so.MAJOR.
    ( cd "$CUDATK/targets/sbsa-linux/lib" 2>/dev/null || exit 0
      for f in *.so.*; do
        case "$f" in *.a) continue;; esac
        b=${f%%.so.*}; rest=${f#*.so.}; maj=${rest%%.*}
        [ -e "$b.so.$maj" ] || ln -sf "$f" "$b.so.$maj"
        [ -e "$b.so" ] || ln -sf "$f" "$b.so"
      done )
  fi
  need_ok "$CUDATK" && TKLIB="$CUDATK"
fi
[ -n "$TKLIB" ] || { echo "FATAL: no COMPLETE CUDA toolkit"; exit 38; }
# cmake finds nvcc via PATH, not CUDAToolkit_ROOT alone.
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "CUDAToolkit_ROOT=$TKLIB"
nvcc --version | tail -2

say "=== [B] source, and the PRECONDITION that it carries both arms ==="
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/pixab-src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 31; }
WANT_SHA=$(cat "$W/pixab-src.sha" 2>/dev/null)
echo "  built_from=$WANT_SHA"
echo "source_sha=$WANT_SHA" >> "$OUT/PROVENANCE"
# BOTH sides, because a half-applied tree satisfies either alone: the swapped op
# without the knob makes the naive arm a second flash arm, and the knob without
# the swap makes the flash arm a second naive one. Either way the A/B renders one
# configuration three times and still prints three columns.
NEWOP=$(grep -c 'vt::AttentionDenseFlash' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
KNOB=$(grep -c 'VLLM_LTX2_DIT_FLASH_ATTN' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
echo "  AttentionDenseFlash call sites: $NEWOP (want >= 1)"
echo "  A/B knob sites:                 $KNOB (want >= 1)"
[ "$NEWOP" -ge 1 ] || { echo "FATAL: #1549 is NOT in this source tree"; exit 40; }
[ "$KNOB"  -ge 1 ] || { echo "FATAL: the A/B knob is NOT in this source tree; both arms would be one arm"; exit 41; }
CMP="$SRC/scripts/ltx25-render-compare.py"
[ -s "$CMP" ] || { echo "FATAL: the comparison tool is not in this source tree"; exit 43; }

say "=== [C] cutlass (resolved, never fetched) ==="
CUT=""
for c in /cutlass /workspace/cutlass /root/cutlass; do
  [ -f "$c/include/cutlass/cutlass.h" ] && CUT="$c" && break
done
TB=/workspace/cutlass-v4.5.0.tar.gz
if [ -z "$CUT" ] && [ -f "$TB" ]; then
  say "unpacking staged cutlass"
  mkdir -p /root/cutlass && tar xzf "$TB" -C /root/cutlass && CUT=/root/cutlass
fi
[ -n "$CUT" ] && [ -f "$CUT/include/cutlass/cutlass.h" ] || { echo "FATAL: no CUTLASS tree"; exit 36; }
say "CUTLASS_DIR=$CUT"

say "=== [D] configure + build (-j 4, per the GB10 recipe) ==="
# The binary is REUSED only when the cached source sha matches the tarball's, so
# a resumed run can never render a tree other than the one it claims. That is
# the failure this guard prevents, not the rebuild.
CACHE="$W/pixab-bin"
BIN=/root/pixabbin; mkdir -p "$BIN"
SKIP_BUILD=0
if [ -s "$CACHE/ltx2-gen" ] && [ -s "$CACHE/libvllm.so.0.0.3" ] && \
   [ -n "$WANT_SHA" ] && [ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SHA" ]; then
  say "REUSING the staged binary: cached SRC_SHA matches $WANT_SHA"
  cp -f "$CACHE/ltx2-gen" "$BIN"/ && chmod 0755 "$BIN/ltx2-gen"
  cp -f "$CACHE/libvllm.so.0.0.3" "$BIN"/ && chmod 0755 "$BIN/libvllm.so.0.0.3"
  cp -f "$CACHE/ltx2_gemma4_text_config.json" "$BIN"/ 2>/dev/null || \
    cp -f "$FULL/bin/ltx2_gemma4_text_config.json" "$BIN"/
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  [ -s "$CACHE/test_ltx2_device" ] && cp -f "$CACHE/test_ltx2_device" "$BIN"/ && chmod 0755 "$BIN/test_ltx2_device"
  SKIP_BUILD=1
fi
if [ "$SKIP_BUILD" = 0 ]; then
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
        -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" > /root/cfg.log 2>&1
  CFG=$?; echo "CFG_RC=$CFG"
  grep -iE 'CUDA target arch|cutlass|flashattention' /root/cfg.log | head -8
  if [ "$CFG" != 0 ]; then
    awk '/CMake (Error|Warning)/,/^$/' /root/cfg.log | head -60; tail -40 /root/cfg.log
    cp -f /root/cfg.log "$OUT/configure-fail.log"; echo "FATAL: configure failed"; exit 33
  fi
  cp -f /root/cfg.log "$OUT/configure.log"
  # Unconstrained parallelism has OOM-rebooted this box.
  ninja -C "$BLD" -j 4 ltx2-gen test_ltx2_device > /root/build.log 2>&1
  B=$?; echo "BUILD_RC=$B"
  echo "  compile_errors=$(grep -ciE ' error: ' /root/build.log)"
  tail -15 /root/build.log
  cp -f /root/build.log "$OUT/build.log"
  [ "$B" = 0 ] || { echo "FATAL: build failed"; exit 34; }
  GEN=$(find "$BLD" -name ltx2-gen -type f | head -1)
  LIB=$(find "$BLD" -name 'libvllm.so.0.0.3' | head -1)
  [ -n "$GEN" ] && [ -n "$LIB" ] || { echo "FATAL: build artefacts not found"; exit 35; }
  cp -f "$GEN" "$BIN"/ && chmod 0755 "$BIN/ltx2-gen"
  cp -f "$LIB" "$BIN"/ && chmod 0755 "$BIN/libvllm.so.0.0.3"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  cp -f "$SRC/tests/vllm/models/ltx2_gemma4_text_config.json" "$BIN"/ 2>/dev/null || \
    cp -f "$FULL/bin/ltx2_gemma4_text_config.json" "$BIN"/
  T=$(find "$BLD" -name test_ltx2_device -type f | head -1)
  [ -n "$T" ] && cp -f "$T" "$BIN"/ && chmod 0755 "$BIN/test_ltx2_device"
  mkdir -p "$CACHE"
  cp -f "$BIN/ltx2-gen" "$CACHE"/ && cp -f "$BIN/libvllm.so.0.0.3" "$CACHE"/
  cp -f "$BIN/ltx2_gemma4_text_config.json" "$CACHE"/ 2>/dev/null
  [ -s "$BIN/test_ltx2_device" ] && cp -f "$BIN/test_ltx2_device" "$CACHE"/
  echo "$WANT_SHA" > "$CACHE/SRC_SHA"
  say "staged the binary for a resumed run"
fi
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
say "ONE BINARY, all three renders: sha256=$BINSHA"
{ echo "binary_sha256=$BINSHA"; echo "binary_built=$([ "$SKIP_BUILD" = 1 ] && echo cache || echo in-lease)"; } >> "$OUT/PROVENANCE"
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
# sha256 and ldd both pass on a file with no execute bit. Ask the binary itself.
"$BIN/ltx2-gen" --help >/dev/null 2>&1 || { echo "FATAL: ltx2-gen will not exec (126 = no exec bit, 127 = missing lib)"; ldd "$BIN/ltx2-gen" | head; exit 25; }
say "EXECUTABLE_OK"

say "=== [E] checkpoints, staged to LOCAL disk ==="
# Measured: 589-1446 s per load over CIFS at 34-83 MiB/s, against ~32 s from
# local disk. Three renders pay that three times, so the ~9 minute copy is not
# an optimisation, it is most of the difference between fitting in a lease and
# not. Each file is matched on EXACT BYTE SIZE, so a half-written stage is
# refused rather than loaded.
declare -A WANT=(
  [ltx-2.5-22b-dev-transformer-bf16.safetensors]=42018190584
  [ltx-2.5-video-vae-conv-bf16.safetensors]=1452269922
  [ltx-2.5-audio-vae-bf16.safetensors]=364866540
  [gemma4-12b-with-proj-nvfp4-torchao.safetensors]=7423624178
)
FREE_K=$(df -k --output=avail /root | tail -1)
NEED_K=$(( (42018190584 + 1452269922 + 364866540 + 7423624178) / 1024 + 8388608 ))
say "local free ${FREE_K}K, need ${NEED_K}K"
CKUSE=$CK
if [ "$FREE_K" -le "$NEED_K" ]; then
  say "NOT staging (insufficient local disk); reading weights over CIFS"
  CKUSE=$FULL/ckpt
else
  for f in "${!WANT[@]}"; do
    s=$FULL/ckpt/$f; d=$CK/$f; want=${WANT[$f]}
    got=$(stat -c %s "$s" 2>/dev/null || echo 0)
    [ "$got" = "$want" ] || { echo "FATAL: source $f is $got bytes, want $want"; exit 23; }
    if [ -s "$d" ] && [ "$(stat -c %s "$d")" = "$want" ]; then say "  already staged $f"; continue; fi
    t=$SECONDS; cp "$s" "$d" || { echo "FATAL: cannot stage $f"; exit 23; }
    [ "$(stat -c %s "$d")" = "$want" ] || { echo "FATAL: short stage of $f"; exit 23; }
    say "  staged $f $want bytes in $((SECONDS-t))s"
  done
fi
# THE FULL/DEV TRANSFORMER, 42,018,190,584 B. Never the distilled one: nothing
# validates checkpoint class (#1137) and it would render plausibly in the wrong
# regime.
say "checkpoints from $CKUSE"
echo "checkpoint_dir=$CKUSE" >> "$OUT/PROVENANCE"

say "=== [F] CORRECTNESS FIRST: the CUDA unit gate, before any render ==="
# AGENTS.md: establish the correctness gate before accepting a performance
# result. `assertions: 0` is a skip wearing a pass and a thrown case shows up
# only on the `Status:` line, so both are printed rather than a grep for ok.
if [ -x "$BIN/test_ltx2_device" ]; then
  "$BIN/test_ltx2_device" > "$OUT/test_ltx2_device.log" 2>&1
  echo "  test_ltx2_device_RC=$?"
  grep -E 'assertions:|test cases:|Status:|SKIP' "$OUT/test_ltx2_device.log" | tail -8
else
  echo "  MISSING test_ltx2_device"
fi

say "=== [G] the three renders ==="
# The prompt, seed and geometry of the recorded 49-frame baseline render
# (out/20260820T223701Z/768x448-49f/render.log line 1), copied byte-for-byte so
# that this pair is additionally comparable to it. The primary evidence is the
# same-binary pair below; the older render was built from a50c57d69, which is an
# ANCESTOR of the swap and therefore a different binary lineage, so it is a
# cross-check and never the control.
PROMPT='A golden retriever shakes water from its coat on a sunlit lawn, droplets flying outward in a bright arc around its head and shoulders, wet fur rippling and separating into strands from shoulders to tail, ears flapping, muscles moving under the coat. Crisp midday light, shallow depth of field, vivid green grass behind. The dog barks once, water patters onto the grass, and a light breeze moves through the trees.'
FRAMES=${FRAMES:-49}; WW=${WW:-768}; HH=${HH:-448}; SEED=${SEED:-20260820}
TOK=$(( (WW/32) * (HH/32) * (((FRAMES-1)/8) + 1) ))
MEM_FLOOR_GIB=${MEM_FLOOR_GIB:-8.0}
say "geometry ${WW}x${HH}/${FRAMES}f = $TOK video tokens, seed $SEED"
say "MemAvailable floor ${MEM_FLOOR_GIB} GiB (the recorded baseline's low-water was 40.13 GiB)"
{
  echo "geometry=${WW}x${HH}/${FRAMES}f"
  echo "video_tokens=$TOK"
  echo "seed=$SEED"
  echo "prompt_sha256=$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
} >> "$OUT/PROVENANCE"

render() {  # $1 = label, $2 = knob value, $3 = hard timeout seconds
  local label=$1 knob=$2 tmo=$3
  local d="$OUT/$label"; mkdir -p "$d"
  local log="$d/render.log"
  say "--- render $label (VLLM_LTX2_DIT_FLASH_ATTN=$knob, hard cap ${tmo}s) ---"
  : > "$log"
  # EVERY RENDER STATES ITS OWN INVOCATION on line 1 of its own log, the way the
  # recorded baseline's render.log does and the way the withdrawn 7.680 s arm did
  # not. An arm whose log cannot say what it ran is not evidence, whatever number
  # it contains.
  {
    echo "[arm] label=$label knob=$knob tmo=${tmo}s"
    echo "[arm] harness=$0 sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
    echo "[arm] binary=$BIN/ltx2-gen sha256=$BINSHA src_sha=$WANT_SHA"
    echo "[arm] geometry=${WW}x${HH}/${FRAMES}f tokens=$TOK seed=$SEED ckpt=$CKUSE"
    echo "[arm] prompt=<<$PROMPT>>"
    echo "[arm] cmd: $BIN/ltx2-gen --pipeline-kind one_stage --dit $CKUSE/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
         "--video-vae $CKUSE/ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae $CKUSE/ltx-2.5-audio-vae-bf16.safetensors" \
         "--checkpoint-class full --encoder $CKUSE/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
         "--encoder-config $BIN/ltx2_gemma4_text_config.json --prompt <see above>" \
         "--frames $FRAMES --width $WW --height $HH --seed $SEED --device cuda" \
         "--workdir $d --out $d/video.mp4"
  } >> "$log"
  # VT_OP_PROVIDER_STATS=1 makes each op announce itself once when it resolves:
  # op=18 is kAttention, op=21 is kAttentionDenseFlash, device=1 is kCUDA.
  # Without it the only evidence of which arm ran is the wall clock, which is one
  # of the things being measured.
  export VLLM_LTX2_DIT_FLASH_ATTN="$knob"
  VT_OP_PROVIDER_STATS=1 timeout -s INT "$tmo" stdbuf -oL -eL "$BIN/ltx2-gen" \
    --pipeline-kind one_stage \
    --dit "$CKUSE/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
    --video-vae "$CKUSE/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae "$CKUSE/ltx-2.5-audio-vae-bf16.safetensors" \
    --checkpoint-class full \
    --encoder "$CKUSE/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
    --encoder-config "$BIN/ltx2_gemma4_text_config.json" \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "$WW" --height "$HH" --seed "$SEED" \
    --device cuda --workdir "$d" --out "$d/video.mp4" >> "$log" 2>&1 &
  local pid=$!
  # MEMORY FLOOR ONLY. No sample cap: the point of this run is a COMPLETED
  # render, and a cap is what left the previous attempt with no frames. No stack
  # sampler either: it would ptrace-stop the process and move the very per-forward
  # times the same run is reducing.
  local stopped_by="none" tick=0
  while kill -0 "$pid" 2>/dev/null; do
    local n avail
    n=$(grep -c 'last=' "$log" 2>/dev/null | head -1)
    avail=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null)
    echo "$(date -u +%H:%M:%S)	$label	forwards=$n	memavail_gib=$avail" >> "$d/watch.tsv"
    # HEARTBEAT ON STDOUT, every ~2 minutes. The engine writes to its own log, so
    # without this the job produces NOTHING on stdout for up to two hours during
    # the naive render, and a lease with an idle timeout would kill a healthy run
    # that is doing exactly what it was asked to do. It doubles as progress: a
    # forward count that stops advancing is visible before the deadline, not
    # after it.
    tick=$((tick + 1))
    [ $((tick % 12)) = 1 ] && say "  [$label] forward $n, MemAvailable ${avail} GiB"
    if [ -n "$avail" ] && awk -v a="$avail" -v f="$MEM_FLOOR_GIB" 'BEGIN{exit !(a<f)}'; then
      stopped_by="memory-floor"; kill -INT "$pid" 2>/dev/null; break
    fi
    sleep 10
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  unset VLLM_LTX2_DIT_FLASH_ATTN
  local nf; nf=$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)
  say "render $label exit=$rc stopped_by=$stopped_by frames=$nf"
  # THE TWO-SIDED ROUTING PROOF, per arm, from that arm's own log. One-sided
  # counting cannot tell a routed call from an added one: the flash arm must show
  # op=21 AND NOT op=18, and the naive arm the reverse. LTX's cross-attentions use
  # op=19 in both, which is why it is printed rather than asserted on.
  echo "--- op-provider selections (18 kAttention / 19 kAttentionCross / 21 kAttentionDenseFlash, device=1 CUDA) ---" | tee -a "$d/ARM"
  grep -E 'op-provider.*op=(18|19|20|21) device=1' "$log" | sort -u | sed 's/^/  /' | tee -a "$d/ARM"
  local n18 n21
  n18=$(grep -cE 'op-provider.*op=18 device=1' "$log")
  n21=$(grep -cE 'op-provider.*op=21 device=1' "$log")
  echo "  op18_naive=$n18 op21_flash=$n21" | tee -a "$d/ARM"
  case "$knob" in
    0) [ "$n18" -ge 1 ] && [ "$n21" = 0 ] && echo "  ROUTING_OK=naive" || echo "  ROUTING_BAD=naive (want op18>=1 op21==0)";;
    *) [ "$n21" -ge 1 ] && [ "$n18" = 0 ] && echo "  ROUTING_OK=flash" || echo "  ROUTING_BAD=flash (want op21>=1 op18==0)";;
  esac | tee -a "$d/ARM"
  # Per-forward MEDIAN from the engine's own `last=` lines. Never the governor,
  # which has reported 1.00 s, 69.1 s, 162 s and 396.9 s for this one quantity.
  grep -ohE 'last=[0-9.]+s' "$log" | sed 's/last=//;s/s$//' > "$d/samples.txt"
  sort -n "$d/samples.txt" | awk -v L="$label" '
    {a[NR]=$1; s+=$1}
    END{ if(!NR){print "  " L ": NO SAMPLES"; exit}
         m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2;
         printf "  %s: n=%d median=%.3fs mean=%.3fs min=%.3fs max=%.3fs\n", L, NR, m, s/NR, a[1], a[NR] }' | tee -a "$d/ARM"
  echo "  memavail low-water: $(awk -F'\t' '{gsub(/memavail_gib=/,"",$4); print $4}' "$d/watch.tsv" 2>/dev/null | sort -n | head -1) GiB" | tee -a "$d/ARM"
  echo "  frames=$nf audio=$([ -s "$d/audio.wav" ] && stat -c %s "$d/audio.wav" || echo 0)" | tee -a "$d/ARM"
}

render flash     1 "${TMO_FLASH:-3600}"
render naive     0 "${TMO_NAIVE:-10800}"
render flash-ctl 1 "${TMO_FLASH:-3600}"

say "=== [H] the speed pair, same binary, same lease, neither arm sampled ==="
python3 - "$OUT/flash/samples.txt" "$OUT/naive/samples.txt" "$OUT/flash-ctl/samples.txt" <<'PY'
import sys, statistics
def med(p):
    try: v=[float(x) for x in open(p).read().split()]
    except OSError: v=[]
    return v, (statistics.median(v) if v else None)
f,fm = med(sys.argv[1]); n,nm = med(sys.argv[2]); c,cm = med(sys.argv[3])
for lab,v,m in (("flash",f,fm),("naive",n,nm),("flash-ctl",c,cm)):
    print(f"  {lab}: n={len(v)} median={m}")
if fm and nm:
    print(f"  SPEEDUP (naive median / flash median) = {nm/fm:.3f}x")
else:
    print("  INCOMPLETE: an arm produced no samples; report that, do not impute")
PY

say "=== [I] the pixel comparison ==="
# The tool is the one committed in this same source tree, run from the tree, so
# the thresholds it applies are the ones the spec derives and not a copy that
# drifted. Its exit status is the gate: 0 pass, 1 a threshold failed, 2 an input
# could not be read. A 2 is never a pass.
python3 "$CMP" \
  --a "$OUT/naive" --b "$OUT/flash" --control "$OUT/flash-ctl" \
  --label-a naive --label-b flash --label-control flash-ctl \
  --json "$OUT/pixel-compare.json" 2>&1 | tee "$OUT/pixel-compare.txt"
echo "PIXEL_COMPARE_RC=${PIPESTATUS[0]}"

say "=== [J] the cross-check against the recorded 20260820 baseline ==="
# A different binary lineage (a50c57d69, an ancestor of the swap), so this is
# never the control and never the A/B. It answers one narrow question: whether a
# naive render is reproducible ACROSS builds, which bounds how much of any delta
# above could be everything-else-on-main rather than the kernel.
OLD=$FULL/out/20260820T223701Z/768x448-49f
if [ -d "$OLD" ]; then
  python3 "$CMP" --a "$OLD" --b "$OUT/naive" \
    --label-a baseline-20260820 --label-b naive \
    --json "$OUT/cross-check.json" 2>&1 | tee "$OUT/cross-check.txt"
  echo "CROSS_CHECK_RC=${PIPESTATUS[0]}"
else
  echo "  the recorded baseline is not on this share; cross-check SKIPPED"
fi

say "=== [K] artefacts ==="
for d in "$OUT"/flash "$OUT"/naive "$OUT"/flash-ctl; do
  [ -d "$d" ] || continue
  printf "  %-24s frames=%s audio=%s mp4=%s\n" "$(basename "$d")" \
    "$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)" \
    "$([ -s "$d/audio.wav" ] && echo yes || echo no)" \
    "$([ -s "$d/video.mp4" ] && echo yes || echo no)"
done
say "DONE OUT=$OUT"
