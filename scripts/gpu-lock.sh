#!/usr/bin/env bash
# The one sanctioned way to take the GPU. Serialise, then prove it.
#
#   scripts/gpu-lock.sh [options] -- <command> [args...]
#
# Options:
#   --lock PATH        lock file (default: $GPU_LOCK, else $HOME/gpu.lock)
#   --timeout SECONDS  bounded wait (default: $GPU_LOCK_TIMEOUT, else 1800)
#   --label TEXT       free-text tag stamped into the record and the sidecar
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
# CANONICAL PATH: $HOME/gpu.lock. It is what .agents/environment.md and
# .agents/benchmarking.md document, it is what most callers take, and it is
# per-user. A lock under /tmp is not clobberable by a stranger -- /tmp is
# sticky, so a non-owner cannot unlink or replace another user's file -- but any
# process on the box can still TAKE the flock, or pre-create the file with modes
# that lock its owner out, and /tmp is cleared on reboot, so the evidence of who
# held it does not survive the OOM-reboot the lock exists to prevent. The env
# override $GPU_LOCK (the key already in .env.example) exists for a host whose
# profile genuinely differs; it is stamped, alongside `lock-source`, so a
# divergence is visible in the record instead of invisible until two jobs
# collide.
#
# THE SWEEP IS STILL OWED, AND #587 IS THEREFORE STILL OPEN. A third spelling
# `/tmp/gpu`, with no extension, is taken as `exec 9>/tmp/gpu` by
# scripts/dgx-online-serving.sh and scripts/dgx-gdn-packed-component.sh, named
# by scripts/opt-dgx-gate.sh and scripts/dgx-sglang-low-concurrency.sh, run by
# .github/workflows/triton-aot-sync.yml, documented as the DGX profile's
# ${GPU_LOCK} in .agents/coordination.md, and used as live gate instructions in
# 53 files under .agents/specs/. The developer's own `.env` sets
# `GPU_LOCK=/tmp/gpu`, and .env.example documents `set -a; . ./.env; set +a`, so
# an agent that loads .env resolves a DIFFERENT file from one that does not --
# through this wrapper. Those callers are outside this change's authority; until
# they are repointed, a benchmark taken here does NOT exclude an online-serving
# or gdn-packed-component run, which is #587 with different filenames. See
# .agents/specs/gpu-lock-wrapper.md for the full enumeration.
#
# THREE PROPERTIES, ALL LEARNED THE HARD WAY.
#
# 1. It REFUSES, never falls back. If the lock cannot be taken, or cannot be
#    PROVEN taken, the wrapper aborts loudly and non-zero and does not run the
#    command. A wrapper that proceeds on the assumption it holds a lock it never
#    took is one more instrument that cannot report its own failure -- the class
#    of defect that produced stale-green builds from an ENOSPC that left the old
#    binary in place, and a policy "violation" that was really a checker unable
#    to write a temp file.
#
# 2. It RECORDS, and it records BEFORE it can refuse. The lock is the mechanism;
#    the stamp is what makes a number defensible afterwards. The wrapped
#    command's exit code comes FIRST in the release block, ahead of disk and
#    load, because disk and load say the box was unhealthy while the exit code
#    says WHICH thing killed the run: 137 (SIGKILL -- another agent's pkill)
#    versus a compiler diagnostic versus an ENOSPC line are three different
#    diagnoses and three different repairs, and only the first says it was not
#    your code at all.
#
#    Stamping does not prevent contention. It prevents contention being
#    undetectable afterwards. A peer's same-binary A/B on this box survived
#    #587 only because its conclusion rested on `diff -r -q` over two output
#    directories -- byte-identity is contention-immune -- while the wall times
#    printed beside it were not. Had the conclusion rested on the times, #587
#    would have voided it silently and nothing in the record would have said so.
#
# 3. Its OWN death releases the lock and says so. A wrapper killed mid-run used
#    to leave ACQUIRE with no RELEASE while the wrapped job kept running through
#    the descriptor it inherited -- so `flock -n` reported the lock still held
#    and the stamped holder-pid named a process that no longer existed. Signals
#    are trapped, forwarded to the job, and answered with a real RELEASE block.
#
# NESTING. The wrapper exports VLLM_CPP_GPU_LOCK_HELD with the resolved path it
# took and VLLM_CPP_GPU_LOCK_PID with the PID holding it. A nested invocation on
# that same path passes through instead of waiting on a lock its own parent
# holds, which would deadlock until the timeout -- but ONLY once the recorded
# PID is shown to be a live ancestor of this process. The variable proves the
# environment NAMES a path; only descent from the live holder proves the
# descriptor that holds the lock is still open in a process we came from.
# Without that check the guard was defeated by any environment that merely named
# the path, and -- non-adversarially -- by any detached job: `subprocess`
# defaults to close_fds=True, so the exported variable reaches the job and the
# locked descriptor does not. The outer wrapper exits, the lock is genuinely
# free, an unrelated job legitimately takes it, and the detached job then stamps
# mode=PASS-THROUGH and runs beside the holder. Two jobs, one lock file, each
# believing it owns the GPU. That claim is now refused (78) instead.
#
# SIDECAR. While the lock is held, `<lock>.holder` names who holds it, why, and
# since when. It exists for the four-hour queue where the only way to tell a
# live measurement from an abandoned server was nvidia-smi archaeology:
#   sidecar + live PID              -> a ten-second conversation with its owner
#   sidecar + dead PIDs             -> a crashed hold, safe to break
#   no sidecar + lock held          -> a pre-wrapper holder, treat as opaque
# Both the wrapper PID and the wrapped job's PID are recorded, because a
# SIGKILLed wrapper cannot trap anything and leaves its descendant holding the
# lock through the inherited descriptor.
#
# Guarantees are pinned by tests/scripts/test_gpu_lock.py, including mutations
# that break each one and must be caught.

set -uo pipefail

EXIT_REFUSED=78
EXIT_TIMEOUT=75
EXIT_USAGE=2

LOCK_ARG=""
LOCK_ARG_GIVEN=0
TIMEOUT=""
LABEL=""
RECORD=""
RECORD_OK=0

LOCK_PATH=""
LOCK_REQUESTED=""
LOCK_SOURCE=""
SIDECAR=""
SIDECAR_OK=0
MODE=""
WAITED=""
ACQUIRED_AT=""
RUNNING=0
RELEASE_DONE=0
CHILD=""
JOB_GROUP=0
WRAPPER_SIGNAL=""

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
    --lock) [ "$#" -ge 2 ] || die_usage "--lock needs a path"; LOCK_ARG="$2"; LOCK_ARG_GIVEN=1; shift 2 ;;
    --lock=*) LOCK_ARG="${1#--lock=}"; LOCK_ARG_GIVEN=1; shift ;;
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

# An EXPLICIT empty --lock is a mistake, not a request for the default. It used
# to resolve $HOME/gpu.lock, which on a gate host is the file everyone else
# holds: a silent redirection inside a wrapper whose stated property is that it
# never falls back.
if [ "$LOCK_ARG_GIVEN" -eq 1 ] && [ -z "$LOCK_ARG" ]; then
  die_usage "--lock was given an empty path"
fi

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
  stamp "lock-source=$LOCK_SOURCE"
  stamp "loadavg=$(loadavg)"
  stamp "df-root=$(df_root)"
  stamp "the command did NOT run"
  exit "$EXIT_REFUSED"
}

# -------------------------------------------------------------- the record --
# BEGIN RECORD-OPEN
# FIRST, before anything that can refuse. It shipped below the lock validation,
# so a lock-path refusal -- the commonest one there is, because `.env` and the
# canonical default disagree on this box -- never created the record file at
# all, while a TIMEOUT reached it. The refusal went to stderr, which is exactly
# what a dispatched job discards. Moving this mutation-swaps back is M15.
if [ -n "$RECORD" ]; then
  # A record we cannot write is the instrument that cannot report its own
  # failure, so it refuses here rather than measuring into nowhere.
  if ( : >>"$RECORD" ) 2>/dev/null; then
    RECORD_OK=1
  else
    refuse "record file is not appendable: $RECORD"
  fi
fi
# END RECORD-OPEN

# ------------------------------------------------------- resolve the path ---
# `lock-source` is stamped beside the path so a fallback is never SILENT. An
# empty GPU_LOCK is the live default on any host that copied .env.example
# verbatim, and falling back there is right -- being unable to see it in the
# record afterwards is not.
if [ "$LOCK_ARG_GIVEN" -eq 1 ]; then
  LOCK_REQUESTED="$LOCK_ARG"
  LOCK_SOURCE="--lock"
elif [ -n "${GPU_LOCK:-}" ]; then
  LOCK_REQUESTED="$GPU_LOCK"
  LOCK_SOURCE="GPU_LOCK"
elif [ -n "${HOME:-}" ]; then
  LOCK_REQUESTED="${HOME}/gpu.lock"
  if [ "${GPU_LOCK+set}" = "set" ]; then
    LOCK_SOURCE="default (GPU_LOCK is set but empty)"
  else
    LOCK_SOURCE="default"
  fi
else
  LOCK_SOURCE="none"
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
# BEGIN LOCK-VALIDATE
if [ -e "$LOCK_PATH" ]; then
  [ -f "$LOCK_PATH" ] || refuse "lock path exists but is not a regular file"
  [ -w "$LOCK_PATH" ] || refuse "lock file is not writable: $LOCK_PATH"
else
  LOCK_DIR="$(dirname -- "$LOCK_PATH")"
  [ -d "$LOCK_DIR" ] || refuse "lock directory does not exist: $LOCK_DIR"
  [ -w "$LOCK_DIR" ] || refuse "lock directory is not writable: $LOCK_DIR"
fi
# END LOCK-VALIDATE

# ------------------------------------------------------------- the sidecar --
SIDECAR="${LOCK_PATH}.holder"

sidecar_open() {
  # Diagnostics, never a gate: the lock is already validated writable above, so
  # a sidecar we cannot write is a surprise worth stamping and not worth
  # refusing a correctly held lock over.
  if ( : >"$SIDECAR" ) 2>/dev/null; then
    SIDECAR_OK=1
    {
      printf 'gpu-lock-holder\n'
      printf 'lock-path=%s\n' "$LOCK_PATH"
      printf 'requested-lock=%s\n' "$LOCK_REQUESTED"
      printf 'label=%s\n' "$LABEL"
      printf 'holder-pid=%s\n' "$$"
      printf 'started-utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'host=%s\n' "$(hostname 2>/dev/null || echo unknown)"
      printf 'command=%s\n' "$CMD_QUOTED"
    } >"$SIDECAR" 2>/dev/null || SIDECAR_OK=0
  fi
  if [ "$SIDECAR_OK" -eq 1 ]; then
    stamp "sidecar=$SIDECAR"
  else
    stamp "sidecar=UNWRITABLE ($SIDECAR)"
  fi
}

sidecar_close() {
  [ "$SIDECAR_OK" -eq 1 ] || return 0
  rm -f -- "$SIDECAR" 2>/dev/null || true
  SIDECAR_OK=0
}

# ------------------------------------------------- is the claim verifiable? --
parent_of() {
  local pid="$1" ppid=""
  if [ -r "/proc/$pid/stat" ]; then
    # `comm` may itself contain spaces and parentheses, so everything up to the
    # LAST ')' is discarded; what follows is state then ppid.
    ppid="$(sed -e 's/.*) //' "/proc/$pid/stat" 2>/dev/null | cut -d' ' -f2)"
  fi
  if [ -z "$ppid" ]; then
    ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
  fi
  printf '%s' "$ppid"
}

# A pass-through is legitimate exactly when the process that recorded the lock
# is a LIVE ANCESTOR of this one: only then is the descriptor holding the lock
# demonstrably still open in a process we descend from. Liveness alone is not
# the property (an unrelated holder is alive too) and naming the path is not the
# property at all. Walking the chain proves liveness as a side effect: a dead
# ancestor is reparented away and can never be found.
lock_pid_is_live_ancestor() {
  local target="$1" pid="$$" hops=0 ppid
  case "$target" in
    '' | *[!0-9]*) return 1 ;;
  esac
  [ "$target" -gt 1 ] || return 1
  while [ "$hops" -lt 64 ]; do
    ppid="$(parent_of "$pid")"
    case "$ppid" in
      '' | *[!0-9]*) return 1 ;;
    esac
    [ "$ppid" = "$target" ] && return 0
    [ "$ppid" -le 1 ] && return 1
    pid="$ppid"
    hops=$((hops + 1))
  done
  return 1
}

# --------------------------------------------------------------- release ----
release() {
  local rc="$1"
  [ "$RELEASE_DONE" -eq 1 ] && return 0
  RELEASE_DONE=1
  sidecar_close
  # exit-code FIRST. Everything below it describes the box; this line describes
  # what happened to the job, and it is the one that says whether the box was
  # even the reason.
  banner "RELEASE"
  stamp "exit-code=$rc"
  if [ "$rc" -gt 128 ] && [ "$rc" -lt 192 ]; then
    local sig=$((rc - 128))
    stamp "exit-reason=killed by signal $sig ($(kill -l "$sig" 2>/dev/null || echo unknown))"
  else
    stamp "exit-reason=exited with status $rc"
  fi
  if [ -n "$WRAPPER_SIGNAL" ]; then
    stamp "wrapper-signal=$WRAPPER_SIGNAL"
  fi
  stamp "outcome=$([ "$MODE" = "PASS-THROUGH" ] && echo PASS-THROUGH || echo RAN)"
  stamp "lock-path=$LOCK_PATH"
  stamp "requested-lock=$LOCK_REQUESTED"
  stamp "holder-pid=$$"
  stamp "waited-seconds=$WAITED"
  stamp "elapsed-seconds=$(since "$ACQUIRED_AT" "$(now)")"
  stamp "released-utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  stamp "loadavg-at-exit=$(loadavg)"
  stamp "df-root-at-exit=$(df_root)"
}

on_exit() {
  local rc=$?
  [ "$RUNNING" -eq 1 ] && release "$rc"
  return 0
}

# The wrapper's own death is the case the shipped version could not report. An
# untrapped SIGTERM left ACQUIRE with no RELEASE, and -- measured -- left the
# wrapped job running orphaned with `flock -n` still reporting the lock HELD,
# because the job inherited the descriptor. Forward the signal so the job dies
# with us, then let the EXIT trap emit a real RELEASE. SIGKILL remains
# untrappable by construction; that case is what the sidecar is for.
on_signal() {
  # A just-forked child still carries this handler until it runs its own
  # `trap -`, and the window is between fork and its FIRST command, so nothing
  # the child executes can close it. Landing there, the inherited handler
  # forwards to a `$CHILD` the child's copy never assigned -- it does nothing,
  # the signal is swallowed and the job survives its own stop order. Measured on
  # this box: 2 of 8 signalled runs sat out the whole job on that window.
  # `BASHPID` is the only thing that differs between the two copies; a child
  # that finds itself here resets and takes the signal for real.
  if [ "${BASHPID:-$$}" != "$$" ]; then
    trap - EXIT TERM INT HUP
    kill -s "$1" "${BASHPID}" 2>/dev/null
    return 0
  fi
  WRAPPER_SIGNAL="$1"
  if [ -n "$CHILD" ]; then
    # BOTH, and in this order. The job itself always exists; its process group
    # does not exist YET in the window between the fork and `setsid` running, so
    # a group-only kill silently hits nothing there and the wrapper sits out the
    # whole job -- measured, 6 attempts out of 6. The group kill is still needed
    # because a gate command is usually a script whose real work is a
    # grandchild, and signalling the shell alone leaves that work running,
    # holding the lock through the descriptor it inherited.
    kill -s "$1" "$CHILD" 2>/dev/null || true
    if [ "$JOB_GROUP" -eq 1 ]; then
      kill -s "$1" -- "-$CHILD" 2>/dev/null || true
    fi
  fi
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

# --------------------------------------------------------------- acquire ----
CMD_QUOTED="$(printf '%q ' "$@")"
START="$(now)"

if [ "${VLLM_CPP_GPU_LOCK_HELD:-}" = "$LOCK_PATH" ]; then
  # An ancestor of this process claims to hold exactly this lock. Waiting on it
  # would deadlock until the timeout -- but only if the claim is TRUE.
  if lock_pid_is_live_ancestor "${VLLM_CPP_GPU_LOCK_PID:-}"; then
    MODE="PASS-THROUGH"
    WAITED="0.000"
  else
    refuse "VLLM_CPP_GPU_LOCK_HELD names $LOCK_PATH but VLLM_CPP_GPU_LOCK_PID=${VLLM_CPP_GPU_LOCK_PID:-<unset>} is not a live ancestor of $$: the lock this environment claims is not provably held. Clear it (env -u VLLM_CPP_GPU_LOCK_HELD -u VLLM_CPP_GPU_LOCK_PID) to take the lock for real."
  fi
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
    stamp "requested-lock=$LOCK_REQUESTED"
    stamp "waited-seconds=$WAITED"
    stamp "timeout-seconds=$TIMEOUT"
    stamp "holder=$(fuser -v "$LOCK_PATH" 2>&1 | tr -s ' \n' ' ' || echo unknown)"
    if [ -r "$SIDECAR" ]; then
      stamp "holder-sidecar=$(tr '\n' ' ' <"$SIDECAR")"
    else
      stamp "holder-sidecar=none (a pre-wrapper holder: treat as opaque)"
    fi
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
stamp "lock-source=$LOCK_SOURCE"
stamp "holder-pid=$$"
stamp "waited-seconds=$WAITED"
stamp "timeout-seconds=$TIMEOUT"
stamp "acquired-utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stamp "host=$(hostname 2>/dev/null || echo unknown)"
stamp "label=$LABEL"
stamp "loadavg=$(loadavg)"
stamp "df-root=$(df_root)"
stamp "command=$CMD_QUOTED"

# Only a real ACQUIRE owns the sidecar. A pass-through must neither rewrite its
# ancestor's file nor remove it on the way out -- the ancestor still holds the
# lock the sidecar describes.
if [ "$MODE" = "ACQUIRED" ]; then
  sidecar_open
fi

export VLLM_CPP_GPU_LOCK_HELD="$LOCK_PATH"
export VLLM_CPP_GPU_LOCK_PID="$$"

RUNNING=1

# ------------------------------------------------------------------- run ----
# Backgrounded so a signal reaches this shell while the job runs: bash defers
# trap handling until a FOREGROUND command finishes, which is why the shipped
# version could not answer its own SIGTERM.
#
# Both of the odd-looking parts are load-bearing, and each was MEASURED:
#
#   `exec`   without it bash forks a subshell that then forks the command, so $!
#            names the SUBSHELL. Forwarding a signal there kills the middleman
#            and leaves the real job running orphaned -- the exact defect this
#            block exists to fix, reintroduced one layer down.
#   `setsid` the job gets its OWN process group, so a forwarded signal reaches
#            the WHOLE tree. Without it, `sh -c 'setup; long_job'` dies at the
#            `sh` and leaves `long_job` orphaned, still holding the lock through
#            the inherited descriptor -- measured, and it is what real gate
#            commands look like. `setsid` execs in place (the job is not already
#            a group leader), so `$!` is still the job itself. Absent, as on
#            macOS, the signal is forwarded to the job alone.
#   `<&0`    without an explicit redirection a non-interactive shell reassigns an
#            asynchronous list's stdin to /dev/null, which would silently break
#            every piped caller.
LAUNCH=()
if command -v setsid >/dev/null 2>&1; then
  LAUNCH=(setsid)
  JOB_GROUP=1
fi
#   `trap -` a freshly forked child still carries the PARENT's handlers until it
#            resets them, and this one's handler forwards to a `$CHILD` the
#            child's copy has never assigned -- so it SWALLOWS the signal and
#            survives. Measured: 2 of 6 signalled runs sat out the whole job on
#            that window alone. Resetting before `exec` closes it, and `exec`
#            keeps the PID.
{ trap - EXIT TERM INT HUP; exec ${LAUNCH[@]+"${LAUNCH[@]}"} "$@"; } <&0 &
CHILD=$!

if [ "$SIDECAR_OK" -eq 1 ]; then
  printf 'command-pid=%s\n' "$CHILD" >>"$SIDECAR" 2>/dev/null || true
fi

# A signal that landed between the acquire and the job starting found no CHILD
# to forward to, so it would otherwise be swallowed and the wrapper would sit
# out the whole job it was told to stop.
#
# CONFIRMED, not fired and forgotten. The job may still be inside its own
# fork/exec window, where the delivery can be lost however carefully the handler
# is written -- bash can note a pending signal and then discard it when the
# child's `trap -` clears the handler it was queued against. That window is
# microseconds and unwinnable by construction, so the repair is to check rather
# than to race: re-send until the job is actually gone, bounded to ~100 ms and
# six signals. This path runs ONLY when the stop order arrived before the job
# began, so there is no graceful shutdown in progress to interrupt.
if [ -n "$WRAPPER_SIGNAL" ]; then
  on_signal "$WRAPPER_SIGNAL"
  for _ in 1 2 3 4 5; do
    kill -0 "$CHILD" 2>/dev/null || break
    sleep 0.02
    on_signal "$WRAPPER_SIGNAL"
  done
fi

wait "$CHILD"
RC=$?
# A trap interrupts `wait` and it returns 128+signal without the job having
# finished. The job's OWN 137 is told apart from that by asking whether it is
# still there: `wait` reaps what it reports on, so a live child means the status
# was the interruption, not the exit.
while [ "$RC" -gt 128 ] && kill -0 "$CHILD" 2>/dev/null; do
  wait "$CHILD"
  RC=$?
done
CHILD=""

exit "$RC"
