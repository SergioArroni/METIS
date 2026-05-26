#!/usr/bin/env bash
set -euo pipefail

# Resource watchdog for long METIS runs.
# It monitors RAM/swap/disk and stops matching processes when critical limits are reached.

INTERVAL_SEC="${INTERVAL_SEC:-30}"
CHECK_PATH="${CHECK_PATH:-/home/ubuntu/METIS}"

RAM_WARN_PCT="${RAM_WARN_PCT:-80}"
RAM_CRITICAL_PCT="${RAM_CRITICAL_PCT:-90}"
SWAP_CRITICAL_PCT="${SWAP_CRITICAL_PCT:-95}"

DISK_WARN_MB="${DISK_WARN_MB:-5000}"
DISK_CRITICAL_MB="${DISK_CRITICAL_MB:-3000}"

TARGET_PATTERN="${TARGET_PATTERN:-python -m metis|metis (calibrate|evaluate)|run_metis_}"
TMUX_SESSION="${TMUX_SESSION:-}"
GRACEFUL_TIMEOUT_SEC="${GRACEFUL_TIMEOUT_SEC:-20}"
ONE_SHOT_STOP="${ONE_SHOT_STOP:-1}"

LOG_FILE="${LOG_FILE:-/home/ubuntu/METIS/logs/watchdogs.log}"
mkdir -p "$(dirname "$LOG_FILE")"

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*" | tee -a "$LOG_FILE"
}

usage() {
  cat <<'EOF'
Usage: scripts/watchdogs.sh [options]

Options:
  --path <dir>              Path used to evaluate free disk space.
  --interval <seconds>      Check interval (default from INTERVAL_SEC).
  --ram-critical <percent>  Critical RAM usage percent.
  --disk-critical <mb>      Critical free disk MB.
  --pattern <regex>         pgrep -f regex to select processes to stop.
  --tmux-session <name>     Optional tmux session to stop with the processes.
  --oneshot <0|1>           Exit after first emergency stop (default 1).
  -h, --help                Show this help.

Environment variables can also be used:
  INTERVAL_SEC, CHECK_PATH, RAM_WARN_PCT, RAM_CRITICAL_PCT, SWAP_CRITICAL_PCT,
  DISK_WARN_MB, DISK_CRITICAL_MB, TARGET_PATTERN, TMUX_SESSION,
  GRACEFUL_TIMEOUT_SEC, ONE_SHOT_STOP, LOG_FILE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      CHECK_PATH="$2"
      shift 2
      ;;
    --interval)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --ram-critical)
      RAM_CRITICAL_PCT="$2"
      shift 2
      ;;
    --disk-critical)
      DISK_CRITICAL_MB="$2"
      shift 2
      ;;
    --pattern)
      TARGET_PATTERN="$2"
      shift 2
      ;;
    --tmux-session)
      TMUX_SESSION="$2"
      shift 2
      ;;
    --oneshot)
      ONE_SHOT_STOP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

get_ram_used_pct() {
  free | awk '/^Mem:/ { printf "%.0f", (1 - ($7 / $2)) * 100 }'
}

get_swap_used_pct() {
  free | awk '/^Swap:/ { if ($2 == 0) { print 0 } else { printf "%.0f", ($3 / $2) * 100 } }'
}

get_disk_free_mb() {
  df -Pm "$CHECK_PATH" | awk 'NR==2 { print $4 }'
}

stop_targets() {
  mapfile -t pids < <(pgrep -f "$TARGET_PATTERN" || true)

  filtered=()
  for pid in "${pids[@]:-}"; do
    [[ "$pid" == "$$" ]] && continue
    filtered+=("$pid")
  done

  if [[ ${#filtered[@]} -eq 0 ]]; then
    log "No matching PIDs for pattern: $TARGET_PATTERN"
  else
    log "Stopping processes (TERM): ${filtered[*]}"
    kill -TERM "${filtered[@]}" 2>/dev/null || true

    end=$((SECONDS + GRACEFUL_TIMEOUT_SEC))
    while (( SECONDS < end )); do
      survivors=()
      for pid in "${filtered[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          survivors+=("$pid")
        fi
      done

      if [[ ${#survivors[@]} -eq 0 ]]; then
        break
      fi
      sleep 1
    done

    survivors=()
    for pid in "${filtered[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        survivors+=("$pid")
      fi
    done

    if [[ ${#survivors[@]} -gt 0 ]]; then
      log "Processes still alive after ${GRACEFUL_TIMEOUT_SEC}s. Forcing KILL: ${survivors[*]}"
      kill -KILL "${survivors[@]}" 2>/dev/null || true
    fi
  fi

  if [[ -n "$TMUX_SESSION" ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      log "Stopping tmux session: $TMUX_SESSION"
      tmux kill-session -t "$TMUX_SESSION" || true
    else
      log "tmux session not found: $TMUX_SESSION"
    fi
  fi
}

log "Watchdog started. Path=$CHECK_PATH interval=${INTERVAL_SEC}s RAM warn/crit=${RAM_WARN_PCT}/${RAM_CRITICAL_PCT}% swap_crit=${SWAP_CRITICAL_PCT}% disk warn/crit=${DISK_WARN_MB}/${DISK_CRITICAL_MB}MB pattern='$TARGET_PATTERN' tmux='${TMUX_SESSION:-none}'"

while true; do
  ram_used="$(get_ram_used_pct)"
  swap_used="$(get_swap_used_pct)"
  disk_free_mb="$(get_disk_free_mb)"

  if (( ram_used >= RAM_WARN_PCT || disk_free_mb <= DISK_WARN_MB )); then
    log "WARN resources: RAM=${ram_used}% swap=${swap_used}% disk_free=${disk_free_mb}MB"
  fi

  if (( ram_used >= RAM_CRITICAL_PCT || swap_used >= SWAP_CRITICAL_PCT || disk_free_mb <= DISK_CRITICAL_MB )); then
    log "CRITICAL resources: RAM=${ram_used}% swap=${swap_used}% disk_free=${disk_free_mb}MB"
    stop_targets

    if [[ "$ONE_SHOT_STOP" == "1" ]]; then
      log "Emergency stop executed. Exiting watchdog."
      exit 0
    fi
  fi

  sleep "$INTERVAL_SEC"
done
