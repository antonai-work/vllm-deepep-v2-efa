#!/usr/bin/env bash
#
# Build the vllm-deepep-v2-efa image in one of two modes.
#
#   --mode fast    (default): FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
#                             Published Wave-1 base image. Skips the ~40 min
#                             cold build of CUDA + EFA + NCCL + DeepEP V2.
#   --mode vanilla         : FROM nvidia/cuda:12.9.0-devel-ubuntu24.04.
#                             Inlines the 240 line base stack from
#                             antonai-work/deepep-v2-efa-base verbatim. Fully
#                             offline-reproducible; no GHCR dependency.
#
# Produces a single image that layers (identical across modes):
#   - nvidia/cuda:12.9.0-devel-ubuntu24.04 (Docker Hub public)
#   - aws-efa-installer 1.48.0
#   - aws-ofi-nccl @ 6e504db (source-built)
#   - NCCL 2.30.4 (pip nvidia-nccl-cu13)
#   - DeepEP V2 @ c84dcac + patches 0001-0003 (PR #612)
#   - vLLM @ tlrmchlsmth/vllm:deepep-v2-integration pinned SHA (PR #41183)
#
# Usage:
#   ./docker/build.sh [--mode fast|vanilla] <image-tag> [extra docker args...]
#
# Examples:
#   ./docker/build.sh vllm-deepep-v2-efa:latest
#   ./docker/build.sh --mode fast    vllm-deepep-v2-efa:fast
#   ./docker/build.sh --mode vanilla vllm-deepep-v2-efa:vanilla
#   ./docker/build.sh --mode fast 123456789012.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-$(git rev-parse --short HEAD)
#
# After build, validate with:
#   docker run --rm <image-tag> bash /opt/docker/preflight.sh
#   -> expected: 8/8 checks PASS
#
# To push:
#   docker push <image-tag>
#
set -euo pipefail

BUILD_MODE="fast"
POSITIONAL=()

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      if [ $# -lt 2 ]; then
        echo "ERROR: --mode requires an argument (fast|vanilla)" >&2
        exit 1
      fi
      BUILD_MODE="$2"
      shift 2
      ;;
    --mode=*)
      BUILD_MODE="${1#--mode=}"
      shift
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

case "${BUILD_MODE}" in
  fast|vanilla) ;;
  *)
    echo "ERROR: --mode must be 'fast' or 'vanilla' (got: ${BUILD_MODE})" >&2
    exit 1
    ;;
esac

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
  echo "Usage: $0 [--mode fast|vanilla] <image-tag> [extra docker args...]" >&2
  echo "Example: $0 --mode fast vllm-deepep-v2-efa:fast" >&2
  exit 1
fi

IMAGE_TAG="${POSITIONAL[0]}"
EXTRA_ARGS=("${POSITIONAL[@]:1}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile"

if [ ! -f "${DOCKERFILE}" ]; then
  echo "ERROR: expected Dockerfile at ${DOCKERFILE}" >&2
  exit 1
fi

# Sanity-check all three DeepEP patches are present (vanilla mode applies them
# via the DeepEP fork checkout; fast mode inherits them from the base image.
# Either way we keep the patch files on disk as reviewable artifacts).
for N in 0001 0002 0003; do
  MATCH=$(ls "${REPO_ROOT}/patches/${N}-"*.patch 2>/dev/null | head -1)
  if [ -z "${MATCH}" ]; then
    echo "ERROR: missing patch ${N}-*.patch in ${REPO_ROOT}/patches/" >&2
    exit 1
  fi
done

echo "[build.sh] Dockerfile:    ${DOCKERFILE}"
echo "[build.sh] Build context: ${REPO_ROOT}"
echo "[build.sh] Image tag:     ${IMAGE_TAG}"
echo "[build.sh] Build mode:    ${BUILD_MODE}"
if [ "${BUILD_MODE}" = "fast" ]; then
  echo "[build.sh] Base image:    ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a"
else
  echo "[build.sh] Base image:    nvidia/cuda:12.9.0-devel-ubuntu24.04 (inlined base stack)"
fi
echo "[build.sh] Patches:       $(ls "${REPO_ROOT}/patches/"*.patch 2>/dev/null | wc -l) present (reviewable artifacts)"
echo ""

DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE}" \
  --build-arg "BUILD_MODE=${BUILD_MODE}" \
  -t "${IMAGE_TAG}" \
  "${EXTRA_ARGS[@]}" \
  "${REPO_ROOT}"

echo ""
echo "[build.sh] Built ${IMAGE_TAG} (mode=${BUILD_MODE})"
echo "[build.sh] Validate: docker run --rm ${IMAGE_TAG} bash /opt/docker/preflight.sh"
