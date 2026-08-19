#!/usr/bin/env bash
# A2-Q2b (#810) dgx:gpu0 gate runner — the device `lm_head` on NVFP4 W4A16 g16.
#
# .agents/specs/nemotron-h-a2q2b-realckpt-lmhead.md.
#
# ── WHY THIS FILE EXISTS RATHER THAN A COMMAND LINE ────────────────────────
# The spec's §4 records that this row has lost FOUR GB10 windows to environment
# rather than to code — `121` instead of `121a`, an unconstrained build, a
# CUTLASS fetch with no egress, and an 8h19m outage. Every one of those is a
# precondition that can be checked before the expensive part starts, so they are
# checked here, once, and a violation VOIDS the run loudly instead of producing
# a number nobody can use.
#
# Env:
#   SRC        checkout to build (required)
#   BUILD      build directory (required; put it on /tmp, NOT /workspace — CIFS
#              holds no symlink and the link step fails there)
#   ARCH       CUDA arch, must be 121a on GB10 (default 121a)
#   LOG_ROOT   where to write the run log (required)
#   CKPT       Nemotron-3.5-Lightning NVFP4 snapshot (default from CHECKPOINT_ROOT)
set -uo pipefail

SRC="${SRC:?set SRC}"
BUILD="${BUILD:?set BUILD}"
ARCH="${ARCH:-121a}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT}"
CKPT="${CKPT:-${CHECKPOINT_ROOT:-/usr/local/nas_share/checkpoints}/nemotron-3.5-lightning-30b-nvfp4}"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/a2q2b-$(date -u +%Y%m%dT%H%M%SZ).log"
rc=0

say() { echo "=== $* ===" | tee -a "$LOG"; }

# `run` reports the COMMAND's exit status, never the pipeline's. `cmd | tail`
# reports tail's status, which is 0 essentially always — a whole gate series has
# read green that way in this tree.
run() {
  local name="$1"; shift
  say "$name"
  "$@" >>"$LOG" 2>&1
  local status=$?
  echo "### $name -> exit $status" | tee -a "$LOG"
  [ "$status" -ne 0 ] && rc=1
  return 0
}

say "A2Q2B GATE start $(date -u +%FT%TZ) host=$(hostname) src=$SRC arch=$ARCH"
(cd "$SRC" && git rev-parse HEAD && git status --short) | tee -a "$LOG"

# ── PRECONDITION 1: host headroom, sampled INSIDE the run, not before it ─────
free -g | tee -a "$LOG"
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL:-0}" -lt 60 ]; then
  say "VOID: only ${AVAIL}G available, floor is 60G (spec §4.2). The box reboots \
rather than OOM-killing at gpu_memory_utilization; this is not a failure of the code."
  exit 3
fi

# ── the build ────────────────────────────────────────────────────────────────
# -j 4: unconstrained parallelism has OOM-REBOOTED this box (AGENTS.md).
say "configure"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DVLLM_CPP_MARLIN=ON >>"$LOG" 2>&1
CFG=$?
echo "### configure -> exit $CFG" | tee -a "$LOG"
[ "$CFG" -ne 0 ] && { say "VOID: configure failed"; exit 3; }

# ── PRECONDITION 2: the arch the configure log actually RESOLVED ─────────────
# A `[121]` or a DISABLED line VOIDS the run rather than failing it: the numbers
# would be from a different kernel set than the one this row is about.
if grep -qiE 'marlin.*DISABLED' "$LOG"; then
  say "VOID: the configure log reports Marlin DISABLED; there is no NVFP4 GEMM to gate"
  exit 3
fi
grep -iE 'marlin|121a?\]' "$LOG" | tail -20 | tee -a "$LOG"

say "build (-j 4)"
cmake --build "$BUILD" -j 4 --target test_nemotron_h_moe_device nemotron-h-gen >>"$LOG" 2>&1
B=$?
echo "### build -> exit $B" | tee -a "$LOG"
[ "$B" -ne 0 ] && { say "BUILD FAILED — this IS a code result, not a void"; exit 1; }

# ── LEG 1: the synthetic device gate ─────────────────────────────────────────
# `-tc` filters are NOT used: doctest splits them on commas, and a comma in a
# case name yields `0 cases ran` + `SUCCESS!`. The whole TU runs, and the case
# count is asserted below.
run "LEG1 synthetic device MoE + lm_head" "$BUILD/tests/test_nemotron_h_moe_device" -s

# A run that executed NO cases prints SUCCESS and exits 0. Assert a non-zero
# case count from the summary line rather than trusting the exit status.
if ! grep -qE 'test cases:[[:space:]]+[1-9]' "$LOG"; then
  say "VOID: the device TU reported no executed test cases"
  rc=1
fi
grep -E 'test cases:|assertions:|Status:' "$LOG" | tail -6 | tee -a "$LOG"

# ── LEG 2: the REAL checkpoint, through the PRODUCTION ABI ───────────────────
# This is the leg that enters through the production entry point:
# `vllm_engine_load` -> the paged runner -> ForwardNemotronHForCausalLM ->
# NemotronHPagedForward -> the device lm_head branch. A unit test that calls
# NemotronHDeviceLmHead directly proves the projection; only this proves the
# capability is REACHED.
if [ -d "$CKPT" ]; then
  run "LEG2 real-checkpoint token identity (device lm_head)" \
    "$BUILD/examples/nemotron-h-gen" \
      --model "$CKPT" \
      --golden "$SRC/tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json" \
      --steps 8 --prompts 3 --max-model-len 2048
else
  say "SKIP LEG2: no checkpoint at $CKPT — this is a SKIP with a named reason, never a pass"
  rc=1
fi

say "A2Q2B GATE overall exit: $rc  log: $LOG"
exit "$rc"
