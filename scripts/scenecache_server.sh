#!/usr/bin/env bash
# Local SceneCache / Qwen3-VL server. Run with start, stop, status, or foreground.
# Override SCENECACHE_PYTHON, MODEL_PATH, CUDA_VISIBLE_DEVICES, PORT, HOST,
# MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS, CONTEXT_LENGTH, or CUDA_HOME as needed.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/.runtime"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/logs/server.log"
SCENECACHE_PYTHON="${SCENECACHE_PYTHON:-/home/lipz/miniconda3/envs/scenecache/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/home/hdd4/lpz/Qwen3-VL-8B-Instruct}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# The parent shell on this machine defaults to CUDA 12.4; this env uses cu128.
if [[ "$CUDA_HOME" == /usr/local/cuda-12.4 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
fi
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$(dirname -- "$SCENECACHE_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SGLANG_CPU_WORKERS="${SGLANG_CPU_WORKERS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$RUNTIME_DIR/flashinfer}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_DIR/triton}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

read_server_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    read -r SERVER_PID < "$PID_FILE"
    [[ "$SERVER_PID" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    # Refuse to signal a PID which has since been reused by another program.
    [[ -r "/proc/$SERVER_PID/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$SERVER_PID/cmdline" | grep -q 'sglang.launch_server'
}

run_server() {
    [[ -x "$SCENECACHE_PYTHON" ]] || { echo "Missing Python: $SCENECACHE_PYTHON" >&2; exit 1; }
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
    [[ -x "$CUDACXX" ]] || { echo "Missing CUDA compiler: $CUDACXX" >&2; exit 1; }
    cd "$PROJECT_ROOT"
    # A small explicit token pool avoids allocating most of a shared GPU.
    # mem-fraction-static affects profiling; max-total-tokens caps allocation.
    exec "$SCENECACHE_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --served-model-name Qwen3-VL-8B-Instruct \
        --host "$HOST" --port "$PORT" \
        --dtype bfloat16 \
        --attention-backend flashinfer \
        --mm-attention-backend triton_attn \
        --sampling-backend pytorch \
        --mem-fraction-static "${MEM_FRACTION_STATIC:-0.95}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS:-8192}" \
        --context-length "${CONTEXT_LENGTH:-4096}" \
        --max-running-requests 1 \
        --chunked-prefill-size 8000 \
        --disable-cuda-graph \
        --disable-radix-cache \
        --trust-remote-code
}

case "${1:-status}" in
    foreground)
        mkdir -p "$RUNTIME_DIR/logs"
        run_server
        ;;
    start)
        mkdir -p "$RUNTIME_DIR/logs"
        if read_server_pid; then
            echo "Already running: PID=$SERVER_PID; log=$LOG_FILE"
            exit 0
        fi
        if "$SCENECACHE_PYTHON" -c 'import socket,sys; s=socket.socket(); sys.exit(s.connect_ex((sys.argv[1],int(sys.argv[2]))) == 0)' "$HOST" "$PORT"; then
            :
        else
            echo "Port $HOST:$PORT is already occupied." >&2
            exit 1
        fi
        if [[ -f "$LOG_FILE" ]]; then
            mv -- "$LOG_FILE" "$LOG_FILE.$(date +%Y%m%d-%H%M%S)"
        fi
        nohup setsid "$PROJECT_ROOT/scripts/scenecache_server.sh" foreground > "$LOG_FILE" 2>&1 < /dev/null &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PID_FILE"
        echo "Starting PID=$SERVER_PID on http://$HOST:$PORT"
        echo "Log: $LOG_FILE"
        ;;
    status)
        if read_server_pid; then
            echo "Running: PID=$SERVER_PID"
            curl --noproxy '*' --fail --silent --show-error --max-time 3 "http://$HOST:$PORT/health"
            echo
        else
            echo "Server is stopped."
            exit 1
        fi
        ;;
    stop)
        if read_server_pid; then
            SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')"
            kill -TERM "$SERVER_PID"
            for ((attempt=0; attempt<20; attempt++)); do
                SERVER_STATE="$(ps -o stat= -p "$SERVER_PID" || true)"
                [[ -z "$SERVER_STATE" || "$SERVER_STATE" == Z* ]] && break
                sleep 0.5
            done
            # A stalled compile/request can prevent SGLang's drain from ending.
            # Only signal the isolated process group created by this script.
            if [[ "$SERVER_PGID" == "$SERVER_PID" ]]; then
                kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
            elif kill -0 "$SERVER_PID" 2>/dev/null; then
                kill -KILL "$SERVER_PID" 2>/dev/null || true
            fi
            echo "Stopped SceneCache PID=$SERVER_PID and its workers"
        else
            echo "Server is stopped."
        fi
        rm -f -- "$PID_FILE"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|foreground}" >&2
        exit 2
        ;;
esac
