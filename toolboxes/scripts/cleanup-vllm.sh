#!/usr/bin/env bash
# cleanup-vllm.sh — Kill stale vLLM processes, clear swap, free RAM.
# Run this after a failed vLLM launch to reset the system for another attempt.
# Usage: ./cleanup-vllm.sh [--dry-run]

set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── 1. Kill any stale vLLM processes ──────────────────────────────
VLLM_PIDS=$(pgrep -f "vllm serve" 2>/dev/null || true)
if [[ -n "$VLLM_PIDS" ]]; then
    log "Killing vLLM processes: $VLLM_PIDS"
    if (( DRY_RUN )); then
        echo "[dry-run] Would kill: $VLLM_PIDS"
    else
        echo "$VLLM_PIDS" | xargs sudo kill -9 2>/dev/null || true
        sleep 2
    fi
else
    log "No stale vLLM processes found."
fi

# ── 2. Remove old/failed vLLM containers ──────────────────────────
OLD_CONTAINERS=$(sudo podman ps -aq --filter "name=b70-vllm" 2>/dev/null || true)
if [[ -n "$OLD_CONTAINERS" ]]; then
    log "Removing stale containers: $OLD_CONTAINERS"
    if (( DRY_RUN )); then
        echo "[dry-run] Would remove: $OLD_CONTAINERS"
    else
        echo "$OLD_CONTAINERS" | xargs sudo podman rm -f 2>/dev/null || true
    fi
else
    log "No stale vLLM containers found."
fi

# ── 3. Drop page cache, inodes, and dentries ─────────────────────
log "Dropping caches..."
if (( DRY_RUN )); then
    echo "[dry-run] Would run: echo 3 > /proc/sys/vm/drop_caches"
else
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
fi
sync

# ── 4. Clear swap by disabling + re-enabling ─────────────────────
log "Clearing swap..."
if (( DRY_RUN )); then
    echo "[dry-run] Would run: swapoff -a && swapon -a"
else
    sudo swapoff --all 2>/dev/null || true
    sleep 1
    sudo swapon --all 2>/dev/null || true
fi

# ── 5. Report state ──────────────────────────────────────────────
echo ""
log "=== Post-cleanup state ==="
free -h
echo ""
swapon --show 2>/dev/null || echo "(no swap)"
echo ""
ps aux | grep -iE "vllm|xpu|ze_init" | grep -v grep | head -5 || echo "(none running)"
