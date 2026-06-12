#!/bin/bash
# Rebuild NCCL and save the mask library as a versioned artifact.
#
# Usage:
#   bash rebuild_nccl_libs.sh vN
#
# What it does:
#   1. Builds NCCL from source (../  relative to this script, i.e. /workspace/nccl)
#   2. Copies /workspace/nccl/build/lib/libnccl.so -> /workspace/nccl/libs/vN/libnccl-mask.so
#
# Rules:
#   - Version string must match vN (e.g. v0, v1, v12).
#   - If vN already exists in libs/tag_version.txt, the save is rejected (immutable).
#   - This script does NOT modify tag_version.txt; append manually to mark as packaged.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

VERSION="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_MASK="${NCCL_ROOT}/build/lib/libnccl.so"
LIBS_ROOT="${NCCL_ROOT}/libs"
TAG_FILE="${LIBS_ROOT}/tag_version.txt"

# ─── Helpers ──────────────────────────────────────────────────────────────────

BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

info()  { echo -e "  ${CYAN}•${RESET} $1"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $1"; }
err()   { echo -e "  ${RED}✗ ERROR:${RESET} $1" >&2; }

# ─── Validation ───────────────────────────────────────────────────────────────

if [[ -z "${VERSION}" ]]; then
    echo -e "${BOLD}Usage:${RESET} $0 vN"
    echo ""
    echo "  Rebuild NCCL and save the mask lib to ${LIBS_ROOT}/vN/libnccl-mask.so"
    echo ""
    echo "  Examples:"
    echo "    $0 v0"
    echo "    $0 v3"
    exit 1
fi

if [[ ! "${VERSION}" =~ ^v[0-9]+$ ]]; then
    err "Version must be in the form vN (e.g. v0, v1, v12), got '${VERSION}'"
    exit 1
fi

if [[ -f "${TAG_FILE}" ]] && grep -qx "${VERSION}" "${TAG_FILE}"; then
    err "Version '${VERSION}' already exists in ${TAG_FILE} (immutable)."
    echo ""
    echo "  Existing packaged versions:"
    sed 's/^/    /' "${TAG_FILE}"
    echo ""
    echo "  Choose a new version number."
    exit 1
fi

# ─── Build ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║${RESET} ${BOLD}Rebuild NCCL & Save Mask Lib (${VERSION})${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"

echo ""
echo -e "${GREEN}▶ Building NCCL...${RESET}"
info "Source: ${NCCL_ROOT}"
info "Command: make CXXSTD=-std=c++17 -j$(nproc) src.build"
echo ""

cd "${NCCL_ROOT}"
make CXXSTD=-std=c++17 -j"$(nproc)" src.build

echo ""
ok "Build completed"

# ─── Verify build output ─────────────────────────────────────────────────────

if [[ ! -f "${SRC_MASK}" ]]; then
    err "Build output not found: ${SRC_MASK}"
    echo "  The build may have failed silently. Check the output above."
    exit 1
fi

# ─── Save artifact ───────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}▶ Saving mask lib...${RESET}"

DST_DIR="${LIBS_ROOT}/${VERSION}"
mkdir -p "${DST_DIR}"
cp "${SRC_MASK}" "${DST_DIR}/libnccl-mask.so"

ok "Saved: ${DST_DIR}/libnccl-mask.so"
info "Size: $(du -h "${DST_DIR}/libnccl-mask.so" | cut -f1)"

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}─────────────────────────────────────────────────${RESET}"
info "Library saved to: ${DST_DIR}/"
ls -lh "${DST_DIR}/libnccl-mask.so" | awk '{print "    " $0}'
echo ""
echo -e "  ${YELLOW}NOTE:${RESET} ${TAG_FILE} was not modified."
echo "  To mark ${VERSION} as packaged, run:"
echo "    echo '${VERSION}' >> ${TAG_FILE}"
echo ""
