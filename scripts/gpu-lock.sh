#!/usr/bin/env bash
# The one sanctioned way to take the GPU. Serialise, then prove it.
#
#   scripts/gpu-lock.sh [options] -- <command> [args...]
#
# Options:
#   --lock PATH        lock file (default: $GPU_LOCK, else $HOME/gpu.lock)
#   --timeout SECONDS  bounded wait (default: $GPU_LOCK_TIMEOUT, else 1800)
#   --label TEXT       free-text tag stamped into the record
#   --record FILE      append the stamp here as well as to stderr
#   -h, --help         this text
#
# Exit status:
#   78  REFUSED  the lock could not be taken; the command did NOT run
#   75  TIMEOUT  the bounded wait expired; the command did NOT run
#    2  usage error; the command did NOT run
#   *   the wrapped command's own exit status, unchanged (137 stays 137)
#
# WHY THIS EXISTS (#587). GPU work on dgx was serialised through flock, but two
# different lock FILES were in use -- $HOME/gpu.lock, which the discipline
# documents and most callers take, and /tmp/gpu.lock, which one job took. Two
# jobs holding different files run concurrently while each believes it owns the
# box, and nothing in the flock discipline reveals it: `fuser $HOME/gpu.lock`
# shows an empty waiter list while the other job is mid-run. On GB10 that is an
# OOM-reboot mechanism, because unified memory means a `gpu_memory_utilization`
# reservation is HOST RAM. It also silently voids measurements.
#
# CANONICAL PATH: $HOME/gpu.lock. It has 432 references in this repository
# against zero for /tmp/gpu.lock, it is what .agents/environment.md and
# .agents/benchmarking.md document, and it is per-user -- a lock under /tmp is
# world-writable, which lets any process on the box hold or clobber the mutex
# that guards someone else's measurement, and /tmp is cleared on reboot, so the
# evidence of who held it does not survive the OOM-reboot it was supposed to
# prevent. The env override $GPU_LOCK (the key already in .env.example) exists
# for a host whose profile genuinely differs; it is stamped, so a divergence is
# visible in the record instead of invisible until two jobs collide.
#
# The 2026-08-13 sweep found a THIRD spelling #587 did not name: `/tmp/gpu`,
# with no extension, taken as `exec 9>/tmp/gpu` by scripts/dgx-online-serving.sh
# and scripts/dgx-gdn-packed-component.sh, named by scripts/opt-dgx-gate.sh and
# scripts/dgx-sglang-low-concurrency.sh, and documented as the DGX profile's
# ${GPU_LOCK} in .agents/coordination.md:99. Those callers are outside this
# change's authority and are owed a follow-up: until they are repointed, a
# benchmark taken through this wrapper does NOT exclude an online-serving or
# gdn-packed-component run, which is #587 with different filenames.
#
# TWO PROPERTIES, BOTH LEARNED THE HARD WAY.
#
# 1. It REFUSES, never falls back. If the lock cannot be taken the wrapper
#    aborts loudly and non-zero and does not run the command. A wrapper that
#    proceeds on the assumption it holds a lock it never took is one more
#    instrument that cannot report its own failure -- the class of defect that
#    produced stale-green builds from an ENOSPC that left the old binary in
#    place, and a policy "violation" that was really a checker unable to write
#    a temp file.
#
# 2. It RECORDS. The lock is the mechanism; the stamp is what makes a number
#    defensible afterwards. The wrapped command's exit code comes FIRST in the
#    release block, ahead of disk and load, because disk and load say the box
#    was unhealthy while the exit code says WHICH thing killed the run: 137
#    (SIGKILL -- another agent's pkill) versus a compiler diagnostic versus an
#    ENOSPC line are three different diagnoses and three different repairs, and
#    only the first says it was not your code at all.
#
#    Stamping does not prevent contention. It prevents contention being
#    undetectable afterwards. A peer's same-binary A/B on this box survived
#    #587 only because its conclusion rested on `diff -r -q` over two output
#    directories -- byte-identity is contention-immune -- while the wall times
#    printed beside it were not. Had the conclusion rested on the times, #587
#    would have voided it silently and nothing in the record would have said so.
#
# NESTING. The wrapper exports VLLM_CPP_GPU_LOCK_HELD with the resolved path it
# took. A nested invocation on that same path passes through instead of waiting
# on a lock its own parent holds, which would deadlock until the timeout. A
# nested invocation on a DIFFERENT path is taken for real.
#
# Guarantees are pinned by tests/scripts/test_gpu_lock.py, including mutations
# that break each one and must be caught.

set -uo pipefail

EXIT_REFUSED=78
EXIT_TIMEOUT=75
EXIT_USAGE=2

LOCK_ARG=""
TIMEOUT=""
LABEL=""
RECORD=""
RECORD_OK=0

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

die_usage() {
  printf 'gpu-lock: usage error: %s\n' "$1" >&2
  printf 'gpu-lock: the command did NOT run\n' >&2
  exit "$EXIT_USAGE"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --lock) [ "$#" -ge 2 ] || die_usage "--lock needs a path"; LOCK_ARG="$2"; shift 2 ;;
    --lock=*) LOCK_ARG="${1#--lock=}"; shift ;;
    --timeout) [ "$#" -ge 2 ] || die_usage "--timeout needs seconds"; TIMEOUT="$2"; shift 2 ;;
    --timeout=*) TIMEOUT="${1#--timeout=}"; shift ;;
    --label) [ "$#" -ge 2 ] || die_usage "--label needs text"; LABEL="$2"; shift 2 ;;
    --label=*) LABEL="${1#--label=}"; shift ;;
    --record) [ "$#" -ge 2 ] || die_usage "--record needs a path"; RECORD="$2"; shift 2 ;;
    --record=*) RECORD="${1#--record=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) die_usage "unknown option: $1" ;;
    *) break ;;
  esac
done

[ "$#" -ge 1 ] || die_usage "no command given (use: gpu-lock.sh [options] -- <command>)"

TIMEOUT="${TIMEOUT:-${GPU_LOCK_TIMEOUT:-1800}}"
case "$TIMEOUT" in
  '' | *[!0-9.]* | *.*.* | .) die_usage "--timeout must be a number of seconds: $TIMEOUT" ;;
esac
awk -v t="$TIMEOUT" 'BEGIN { exit !(t + 0 > 0) }' \
  || die_usage "--timeout must be greater than zero: $TIMEOUT"

# ---------------------------------------------------------------- stamping --
stamp() {
  printf 'gpu-lock: %s\n' "$1" >&2
  if [ "$RECORD_OK" -eq 1 ]; then
    printf 'gpu-lock: %s\n' "$1" >>"$RECORD"
  fi
}

banner() {
  stamp "=== $1 ==="
}

now() { date +%s.%N; }

since() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }

loadavg() {
  if [ -r /proc/loadavg ]; then
    cut -d' ' -f1-3 /proc/loadavg
  else
    uptime | sed 's/.*load averages\{0,1\}: //' | tr -d ','
  fi
}

df_root() {
  # `-P` keeps one filesystem per line, so the row cannot wrap and lose the
  # use% that is the whole reason to look. A full disk does not fail loudly.
  df -hP / 2>/dev/null | tail -1 | tr -s ' '
}

# ---------------------------------------------------------------- refusal ---
# The single exit in this function is the anchor for the "refusal falls
# through" mutation. Reaching the end of this function without exiting is the
# defect the wrapper exists to make impossible.
refuse() {
  banner "REFUSED"
  stamp "outcome=REFUSED"
  stamp "reason=$1"
  stamp "lock-path=$LOCK_PATH"
  stamp "requested-lock=$LOCK_REQUESTED"
  stamp "loadavg=$(loadavg)"
  stamp "df-root=$(df_root)"
  stamp "the command did NOT run"
  exit "$EXIT_REFUSED"
}

# ------------------------------------------------------- resolve the path ---
LOCK_REQUESTED="${LOCK_ARG:-${GPU_LOCK:-${HOME:-}/gpu.lock}}"
if [ -z "${LOCK_ARG:-}" ] && [ -z "${GPU_LOCK:-}" ] && [ -z "${HOME:-}" ]; then
  LOCK_PATH=""
  refuse "no lock path: HOME is unset and neither --lock nor GPU_LOCK was given"
fi

resolve() {
  local p="$1" resolved d b
  if resolved="$(readlink -f -- "$p" 2>/dev/null)" && [ -n "$resolved" ]; then
    printf '%s\n' "$resolved"
    return 0
  fi
  d="$(dirname -- "$p")"
  b="$(basename -- "$p")"
  if d="$(cd -- "$d" 2>/dev/null && pwd -P)"; then
    printf '%s/%s\n' "$d" "$b"
    return 0
  fi
  return 1
}

if ! LOCK_PATH="$(resolve "$LOCK_REQUESTED")"; then
  LOCK_PATH="$LOCK_REQUESTED"
  refuse "lock directory does not exist or is unreachable: $(dirname -- "$LOCK_REQUESTED")"
fi

# ------------------------------------------------- validate before stamping --
if [ -e "$LOCK_PATH" ]; then
  [ -f "$LOCK_PATH" ] || refuse "lock path exists but is not a regular file"
  [ -w "$LOCK_PATH" ] || refuse "lock file is not writable: $LOCK_PATH"
else
  LOCK_DIR="$(dirname -- "$LOCK_PATH")"
  [ -d "$LOCK_DIR" ] || refuse "lock directory does not exist: $LOCK_DIR"
  [ -w "$LOCK_DIR" ] || refuse "lock directory is not writable: $LOCK_DIR"
fi

if [ -n "$RECORD" ]; then
  # A record we cannot write is the instrument that cannot report its own
  # failure, so it refuses here rather than measuring into nowhere.
  if ( : >>"$RECORD" ) 2>/dev/null; then
    RECORD_OK=1
  else
    refuse "record file is not appendable: $RECORD"
  fi
fi

# --------------------------------------------------------------- acquire ----
CMD_QUOTED="$(printf '%q ' "$@")"
START="$(now)"

if [ "${VLLM_CPP_GPU_LOCK_HELD:-}" = "$LOCK_PATH" ]; then
  # An ancestor of this process already holds exactly this lock. Waiting on it
  # would deadlock until the timeout, so pass through and say so in the record.
  MODE="PASS-THROUGH"
  WAITED="0.000"
else
  MODE="ACQUIRED"
  if ! exec {LOCK_FD}>>"$LOCK_PATH"; then
    refuse "could not open the lock file for writing: $LOCK_PATH"
  fi
  if ! flock -w "$TIMEOUT" "$LOCK_FD"; then
    WAITED="$(since "$START" "$(now)")"
    banner "TIMEOUT"
    stamp "outcome=TIMEOUT"
    stamp "lock-path=$LOCK_PATH"
    stamp "waited-seconds=$WAITED"
    stamp "timeout-seconds=$TIMEOUT"
    stamp "holder=$(fuser -v "$LOCK_PATH" 2>&1 | tr -s ' \n' ' ' || echo unknown)"
    stamp "loadavg=$(loadavg)"
    stamp "df-root=$(df_root)"
    stamp "the command did NOT run"
    exit "$EXIT_TIMEOUT"
  fi
  WAITED="$(since "$START" "$(now)")"
fi

ACQUIRED_AT="$(now)"

banner "ACQUIRE"
stamp "mode=$MODE"
stamp "lock-path=$LOCK_PATH"
stamp "requested-lock=$LOCK_REQUESTED"
stamp "holder-pid=$$"
stamp "waited-seconds=$WAITED"
stamp "timeout-seconds=$TIMEOUT"
stamp "acquired-utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stamp "host=$(hostname 2>/dev/null || echo unknown)"
stamp "label=$LABEL"
stamp "loadavg=$(loadavg)"
stamp "df-root=$(df_root)"
stamp "command=$CMD_QUOTED"

export VLLM_CPP_GPU_LOCK_HELD="$LOCK_PATH"
export VLLM_CPP_GPU_LOCK_PID="$$"

# ------------------------------------------------------------------- run ----
"$@"
RC=$?

# ---------------------------------------------------------------- release ---
# exit-code FIRST. Everything below it describes the box; this line describes
# what happened to the job, and it is the one that says whether the box was
# even the reason.
banner "RELEASE"
stamp "exit-code=$RC"
if [ "$RC" -gt 128 ] && [ "$RC" -lt 192 ]; then
  SIG=$((RC - 128))
  stamp "exit-reason=killed by signal $SIG ($(kill -l "$SIG" 2>/dev/null || echo unknown))"
else
  stamp "exit-reason=exited with status $RC"
fi
stamp "outcome=$([ "$MODE" = "PASS-THROUGH" ] && echo PASS-THROUGH || echo RAN)"
stamp "lock-path=$LOCK_PATH"
stamp "holder-pid=$$"
stamp "waited-seconds=$WAITED"
stamp "elapsed-seconds=$(since "$ACQUIRED_AT" "$(now)")"
stamp "released-utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stamp "loadavg-at-exit=$(loadavg)"
stamp "df-root-at-exit=$(df_root)"

exit "$RC"
