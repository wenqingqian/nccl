#!/bin/bash
# Run the full mask all-reduce test suite.
#
# Usage:
#   bash run.sh                        # default: 8 GPUs, skip rebuild
#   bash run.sh --rebuild v2           # rebuild version v2 before testing
#   NPROC=4 bash run.sh               # override GPU count
#
# Environment variables:
#   NPROC   - number of GPUs (default: 8)
#   WARMUP  - warmup iterations (default: 50)
#   ITERS   - timed iterations (default: 50)
#
# The script forces NCCL_PROTO=Simple so the mask logic in prims_simple.h is used.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

NPROC="${NPROC:-8}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-50}"
REBUILD_VERSION=""
PROFILE_MODE=0
PROF_SIZE=""
PROF_SPARSITY=""
PROF_DISTRIBUTION=""

export NCCL_PROTO=Simple

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ─── Argument parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --rebuild requires a version argument (e.g. --rebuild v2)"
                exit 1
            fi
            REBUILD_VERSION="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --profile)
            PROFILE_MODE=1
            shift
            ;;
        --size)
            PROF_SIZE="$2"
            shift 2
            ;;
        --sparsity)
            PROF_SPARSITY="$2"
            shift 2
            ;;
        --distribution)
            PROF_DISTRIBUTION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rebuild vN       Rebuild NCCL mask lib version vN before testing"
            echo "  --nproc N          Number of GPUs (default: 8)"
            echo "  --warmup N         Warmup iterations (default: 50)"
            echo "  --iters N          Timed iterations (default: 50)"
            echo "  --profile          Profile-only mode (skip correctness & perf)"
            echo "  --size SIZE        Tensor size (e.g. 64MB). Profile default: 64MB"
            echo "  --sparsity VAL     Sparsity (e.g. 0.5). Profile default: 0.5"
            echo "  --distribution D   Distribution (e.g. block). Profile default: block"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'. Use --help for usage."
            exit 1
            ;;
    esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────

BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║${RESET} ${BOLD}$1${RESET}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
}

step() {
    echo ""
    echo -e "${GREEN}▶ $1${RESET}"
    echo -e "${GREEN}─────────────────────────────────────────────────${RESET}"
}

info() {
    echo -e "  ${CYAN}•${RESET} $1"
}

warn() {
    echo -e "  ${YELLOW}⚠ $1${RESET}"
}

run_torchrun() {
    torchrun --nproc_per_node="${NPROC}" "$@"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

banner "Mask All-Reduce Test Suite"
info "GPUs:          ${NPROC}"
info "Warmup:        ${WARMUP} iterations"
info "Timed iters:   ${ITERS}"
info "NCCL_PROTO:    Simple"

# Optional rebuild step
if [[ -n "${REBUILD_VERSION}" ]]; then
    step "Step 0: Rebuild NCCL mask lib (${REBUILD_VERSION})"
    bash "${SCRIPT_DIR}/rebuild_nccl_libs.sh" "${REBUILD_VERSION}"
fi

# Display sizes of all loaded NCCL libraries.
step "Loaded NCCL libraries"
LIBS_DIR="/workspace/nccl/libs"
if [[ -d "${LIBS_DIR}" ]]; then
    for lib in "${LIBS_DIR}"/nomask/libnccl-nomask.so "${LIBS_DIR}"/v*/libnccl-mask.so; do
        [[ -f "${lib}" ]] || continue
        dir_name=$(basename "$(dirname "${lib}")")
        file_name=$(basename "${lib}")
        size=$(du -h "${lib}" | cut -f1)
        info "${dir_name}/${file_name}: ${size}"
    done
else
    warn "Library directory not found: ${LIBS_DIR}"
fi

# Correctness tests (always run, abort on failure)
step "Step 1: Correctness tests"
if ! run_torchrun test_mask.py correctness; then
    echo -e "${RED}Correctness tests FAILED. Aborting.${RESET}"
    exit 1
fi

if [[ "${PROFILE_MODE}" -eq 1 ]]; then
    # ─── Profile-only mode ───────────────────────────────────────────────────
    # Defaults for prof: test_mask.py prof requires these three params.
    PROF_SIZE="${PROF_SIZE:-64MB}"
    PROF_SPARSITY="${PROF_SPARSITY:-0.5}"
    PROF_DISTRIBUTION="${PROF_DISTRIBUTION:-block}"

    step "Profile (nsys, warmup excluded)"
    info "size:         ${PROF_SIZE}"
    info "sparsity:     ${PROF_SPARSITY}"
    info "distribution: ${PROF_DISTRIBUTION}"
    info "warmup:       ${WARMUP} (untraced)"
    info "iters:        ${ITERS} (profiled)"

    NSYS_OUTPUT="${SCRIPT_DIR}/prof_${PROF_SIZE}_sp${PROF_SPARSITY}_${PROF_DISTRIBUTION}"

    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output="${NSYS_OUTPUT}" \
        --force-overwrite=true \
        torchrun --nproc_per_node="${NPROC}" test_mask.py prof \
            --warmup="${WARMUP}" --iters="${ITERS}" \
            --sizes="${PROF_SIZE}" --sparsities="${PROF_SPARSITY}" \
            --distributions="${PROF_DISTRIBUTION}"

    info "nsys report saved to: ${NSYS_OUTPUT}.nsys-rep"
else
    # ─── Normal mode: perf ───────────────────────────────────────────────────
    step "Step 2: Performance sweep (size × sparsity × distribution)"
    PERF_ARGS=(--warmup="${WARMUP}" --iters="${ITERS}")
    [[ -n "${PROF_SIZE}" ]]         && PERF_ARGS+=(--sizes="${PROF_SIZE}")
    [[ -n "${PROF_SPARSITY}" ]]     && PERF_ARGS+=(--sparsities="${PROF_SPARSITY}")
    [[ -n "${PROF_DISTRIBUTION}" ]] && PERF_ARGS+=(--distributions="${PROF_DISTRIBUTION}")
    run_torchrun test_mask.py perf "${PERF_ARGS[@]}"
fi

# Done
banner "All tests completed successfully ✓"
