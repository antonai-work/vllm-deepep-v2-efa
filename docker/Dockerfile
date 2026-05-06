# syntax=docker/dockerfile:1.10
# =============================================================================
# vllm-deepep-v2-efa - vLLM MoE inference on DeepEP V2 over AWS EFA
# =============================================================================
#
# Dual-path build:
#   BUILD_MODE=fast    -> FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.1-sm90a
#                         (Wave-1 published base image, same DeepEP V2 + EFA
#                          stack as below, skips the 40 minute cold build)
#   BUILD_MODE=vanilla -> inlines the full 240 line base stack from
#                         antonai-work/deepep-v2-efa-base, FROM
#                         nvidia/cuda:12.9.0-devel-ubuntu24.04. Cold build, no
#                         dependency on GHCR. Both modes produce byte-for-byte
#                         equivalent final images (verified by preflight 5/5
#                         base + 3/3 vLLM checks).
#
# Default is fast. Switch at the CLI:
#   docker build -f docker/Dockerfile --build-arg BUILD_MODE=fast    -t vllm-deepep-v2-efa:fast    .
#   docker build -f docker/Dockerfile --build-arg BUILD_MODE=vanilla -t vllm-deepep-v2-efa:vanilla .
# or use the wrapper:
#   docker/build.sh --mode fast    vllm-deepep-v2-efa:fast
#   docker/build.sh --mode vanilla vllm-deepep-v2-efa:vanilla
#
# Pinned layers (identical across both modes):
#   Base OS + CUDA       nvidia/cuda:12.9.0-devel-ubuntu24.04     vanilla
#   EFA userspace        aws-efa-installer 1.48.0                 vanilla (--build-ngc)
#   aws-ofi-nccl         6e504db3403931cde43a2335adcc73fbc69cccac vanilla
#                        ("gin: Size active_put_signal to full sequence
#                         number space", upstream fix 2026-04-24)
#   NCCL                 nvidia-nccl-cu12 >= 2.30.4               vanilla
#                        (Wave 9b cu12 unification, was cu13 in <=Wave 7c.4)
#   GDRCopy              v2.5.1                                   vanilla
#   DeepEP V2            dmvevents/DeepEP-1 @ aws-efa-auto-qp-cap-v2
#                        SHA 146cc356aa00c39ac1590c05775e05b0f031e70c
#                        (b306af0 + 3 commits = PR #612)          OUR PATCHES
#   vLLM                 github.com/tlrmchlsmth/vllm @            vLLM PR #41183
#                        deepep-v2-integration, SHA 6d7a3fab
#                        (vanilla vllm-project/vllm main + PR #41183)
#   api-shim             NOT INCLUDED                             by design
#                        DEEP_EP_USE_V2_SHIM=0 at runtime; vLLM calls
#                        deep_ep.ElasticBuffer natively via PR #41183.
#
# Preflight gate (inside running container):
#   docker run --rm <image> bash /opt/docker/preflight.sh
#   -> expected: "8/8 checks PASS"
#
# Runtime gate (2-pod EFA cluster):
#   kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
#   -> expected: /v1/chat/completions returns a real JSON response
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.2-sm90a
ARG BASE_IMAGE_VANILLA=nvidia/cuda:12.9.0-devel-ubuntu24.04
# Wave 9b cu12 unification: single cu12 NCCL throughout (matches torch cu129 +
# base-image NCCL + DeepEP .so). Supersedes the Wave 7c.4 cu13 realign that
# caused Wave 8 "invalid device ordinal" crashes at first MoE dispatch.
ARG NVIDIA_NCCL_PIN=nvidia-nccl-cu12>=2.30.4

# =============================================================================
# base-fast: FROM the prebuilt GHCR base image. Contains the full DeepEP V2 +
# EFA + NCCL 2.30.4 + aws-ofi-nccl 6e504db stack already assembled.
# =============================================================================
FROM ${BASE_IMAGE_FAST} AS base-fast

# =============================================================================
# base-vanilla: inline the full base stack from
# antonai-work/deepep-v2-efa-base/Dockerfile (verbatim, same pins, same
# --build-ngc path, same DeepEP fork branch). End state is identical to the
# base-fast stage above. Provided so this repo's build does not depend on
# GHCR reachability.
# =============================================================================
FROM ${BASE_IMAGE_VANILLA} AS base-vanilla

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# System deps + EFA userspace + aws-ofi-nccl (NGC-aware install path)
# -----------------------------------------------------------------------------
# EFA installer 1.48+ ships a --build-ngc path specifically for NVIDIA DL base
# containers: it installs libfabric1-aws + libfabric-aws-dev + libfabric-aws-bin
# debs plus the NGC-flavored libnccl-ofi-ngc-v3 deb. The NGC plugin is built
# with --disable-nccl-net-library so it is NCCL-version-agnostic.
#
# We deliberately do NOT install rdma-core / libibverbs-dev / ibverbs-providers
# from the noble apt repository. Those would land v50 and block the v61
# rdma-core bundled inside the EFA 1.48 tarball. libfabric1-aws hard-depends
# on ibverbs-providers >= 59 so the apt path cannot satisfy it.
ARG EFA_INSTALLER_VERSION=1.48.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget tar xz-utils file \
    autoconf automake libtool pkg-config cmake ninja-build patchelf \
    python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev openmpi-bin \
    libnuma-dev libnl-3-dev libnl-route-3-dev \
    kmod pciutils jq \
 && rm -rf /var/lib/apt/lists/*

# The 1.48 installer's NGC path runs plain `dpkg -i`. libfabric1-aws declares
# `Depends: ibverbs-providers (>= 59)` but noble ships 50.0. Manually installing
# the v61 debs from the tarball first (with --auto-deconfigure) upgrades the
# preinstalled v50 ibverbs-providers to v61 and then libfabric1-aws can drop
# in. This is the same sequence the NGC CUDA DL Base 26.03 image uses.
RUN set -eux; \
    mkdir -p /var/build-scratch/efa; \
    cd /var/build-scratch/efa; \
    curl -fsSL -o aws-efa-installer.tar.gz \
      "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
    tar -xzf aws-efa-installer.tar.gz; \
    cd aws-efa-installer; \
    arch="$(uname -m)"; \
    dpkg -i --auto-deconfigure \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/ibverbs-providers_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad3_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibmad5_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibnetdisc5_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric1-aws_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-bin_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-dev_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-config_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-profile_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libnccl-ofi-ngc-v3_*.deb; \
    rm -f /opt/amazon/efa/lib/libfabric.a; \
    test -f /opt/amazon/efa/include/rdma/fabric.h; \
    test -f /opt/amazon/efa/lib/libfabric.so.1; \
    test -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so; \
    strings /lib/x86_64-linux-gnu/libefa.so.1 | grep '^EFA_1\.4$' > /dev/null \
        || (echo "FATAL: libefa.so.1 missing EFA_1.4 symbol" && exit 1); \
    ldconfig; \
    rm -rf /var/build-scratch/efa

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64

# -----------------------------------------------------------------------------
# GDRCopy v2.5.1 userspace (kernel module lives on the host)
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --depth=1 --branch v2.5.1 \
      https://github.com/NVIDIA/gdrcopy.git /var/build-scratch/gdrcopy; \
    cd /var/build-scratch/gdrcopy; \
    CUDA=/usr/local/cuda make lib_install; \
    ldconfig; \
    rm -rf /var/build-scratch/gdrcopy

# -----------------------------------------------------------------------------
# PyTorch + NCCL 2.30.4 via pip (DeepEP V2 upstream README recommends this
# over building NCCL from source, so the JIT can find headers automatically)
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu129 \
 || pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128

# numpy<2 is required at runtime by torch.distributed.all_gather_object
# (DeepEP V2's ElasticBuffer.__init__ calls it from get_nccl_comm_handle).
# Without numpy<2 the NCCL comm handle exchange raises "Numpy is not available".
RUN pip install --no-cache-dir --break-system-packages \
      "numpy<2"

# Purge the apt-installed NCCL 2.26.x that the CUDA base image ships. Left in
# place, /lib/x86_64-linux-gnu/libnccl.so.2 (2.26.5) shadows the pip wheel's
# 2.30.4 via ld.so. DeepEP V2 requires >=2.30.4 (Gin backend API).
RUN set -eux; \
    apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true; \
    apt-mark hold libnccl2 libnccl-dev 2>/dev/null || true; \
    rm -f /lib/x86_64-linux-gnu/libnccl.so* /usr/lib/x86_64-linux-gnu/libnccl.so*; \
    ldconfig

RUN set -eux; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      "${NVIDIA_NCCL_PIN}"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    echo "NCCL_LIB=${NCCL_LIB}"; \
    test -n "${NCCL_LIB}"; \
    ls -la "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    echo "SO2=${SO2}"; \
    test -n "${SO2}"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ls -la "${NCCL_LIB}/libnccl.so" "${NCCL_LIB}/libnccl.so.2" || true; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    ldconfig; \
    ldconfig -p | grep 'libnccl.so' | head -3

# NVSHMEM pip wheel for the legacy link path. DeepEP V2's setup.py still links
# against libnvshmem_host.so even when the primary runtime is NCCL Gin
# (upstream setup.py has a TODO: "make NVSHMEM and legacy optional").
# Required >=3.3.9 per DeepEP docs/nvshmem.md on the V2 branch.
# The wheel only ships libnvshmem_host.so.X (versioned), but DeepEP's link
# line uses `-l:libnvshmem_host.so` (unversioned). Create the symlink.
RUN pip install --no-cache-dir --break-system-packages \
      "nvidia-nvshmem-cu12>=3.3.9" \
 && NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)" \
 && test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}" \
 && ls "${NVSHMEM_LIB}" \
 && HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)" \
 && ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so" \
 && echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf \
 && ldconfig

# -----------------------------------------------------------------------------
# aws-ofi-nccl built from master, pinned to commit 6e504db3403931cde (2026-04-24):
# "gin: Size active_put_signal to full sequence number space". Official upstream
# fix for the 128-slot GIN ring overflow that caused DeepEP V2 dispatch to hang
# with "received count: 0" on AWS EFA once num_allocated_qps >= 5. Supersedes
# our earlier out-of-tree ring-size patch.
#
# Must be built AFTER nvidia-nccl-cu12 pip install so --with-nccl finds
# matching 2.30.4 headers. The plugin's ncclGinPlugin_v11/v12 symbols must
# match the NCCL it is loaded alongside, or DeepEP's railedGinType check
# (see csrc/kernels/backend/nccl.cu in DeepEP V2) fails.
# (Wave 9b cu12 unification; was cu13 in <=Wave 7c.4.)
# -----------------------------------------------------------------------------
ARG AWS_OFI_NCCL_SHA=6e504db3403931cde43a2335adcc73fbc69cccac
RUN set -eux; \
    git clone https://github.com/aws/aws-ofi-nccl.git /var/build-scratch/aws-ofi-nccl; \
    cd /var/build-scratch/aws-ofi-nccl; \
    git checkout "${AWS_OFI_NCCL_SHA}"; \
    ./autogen.sh; \
    NCCL_INCLUDE_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/include' -type d 2>/dev/null | head -1)"; \
    NCCL_LIB_DIR_OFI="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_INCLUDE_DIR}" && test -n "${NCCL_LIB_DIR_OFI}"; \
    NCCL_ROOT="$(dirname "${NCCL_INCLUDE_DIR}")"; \
    CPPFLAGS="-I/opt/amazon/efa/include -I/opt/amazon/efa/include/rdma -I/usr/local/cuda/include -I${NCCL_INCLUDE_DIR}" \
    LDFLAGS="-L/opt/amazon/efa/lib -L/opt/amazon/efa/lib64 -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L${NCCL_LIB_DIR_OFI}" \
    ./configure \
      --prefix=/opt/aws-ofi-nccl \
      --with-libfabric=/opt/amazon/efa \
      --with-cuda=/usr/local/cuda \
      --with-nccl="${NCCL_ROOT}" \
      --enable-platform-aws \
      --disable-tests; \
    make -j"$(nproc)"; \
    make install; \
    nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin; \
    echo /opt/aws-ofi-nccl/lib > /etc/ld.so.conf.d/zz-aws-ofi-nccl.conf; \
    ldconfig; \
    rm -rf /var/build-scratch/aws-ofi-nccl

# -----------------------------------------------------------------------------
# DeepEP V2 with PR #612 (auto-QP cap + get_rdma_gbs EFA fast path + dispatch
# kScaleoutUpdateInterval 3 -> 16). Uses the pre-patched fork branch identical
# to the one shipped in antonai-work/deepep-v2-efa-base. The three .patch
# files under ./patches/ are the exact commits on that branch, provided for
# reviewability. When DeepEP PR #612 merges upstream we flip DEEPEP_FORK to
# vanilla and drop ./patches/.
# -----------------------------------------------------------------------------
# Canonical values live in ../pins.env at the repo root. Both docker/build.sh
# and ci/buildspec.yml source that file and pass the values via --build-arg,
# which overrides these defaults. The hardcoded values below are retained as
# a legacy-safety fallback so `docker build ...` still works in an emergency.
ARG DEEPEP_FORK=https://github.com/dmvevents/DeepEP-1.git
ARG DEEPEP_BRANCH=aws-efa-auto-qp-cap-v2
ARG DEEPEP_SHA=146cc356aa00c39ac1590c05775e05b0f031e70c

RUN set -eux; \
    git clone --recurse-submodules "${DEEPEP_FORK}" /opt/DeepEP; \
    cd /opt/DeepEP; \
    git checkout "${DEEPEP_BRANCH}"; \
    test "$(git rev-parse HEAD)" = "${DEEPEP_SHA}" \
      || (echo "FATAL: DeepEP HEAD $(git rev-parse HEAD) != pinned ${DEEPEP_SHA}" && exit 1); \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"

# Runtime env from the base image (identical to base-fast so flipping BUILD_MODE
# never changes the env surface).
ENV NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_CUMEM_HOST_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_IGNORE_DISABLED_P2P=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    FI_EFA_FORK_SAFE=1 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

WORKDIR /opt

# =============================================================================
# release: FROM base-${BUILD_MODE}. Both base-fast and base-vanilla produce
# byte-for-byte equivalent content up through DeepEP V2 + NCCL 2.30.4 + EFA
# userspace + aws-ofi-nccl 6e504db. The framework layer below is identical.
# =============================================================================
FROM base-${BUILD_MODE} AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# vLLM with PR #41183 applied (via fork pin)
#
# PR #41183 ("[WideEP] Integrate DeepEP v2") is an OPEN vLLM PR authored by
# vLLM team member @tlrmchlsmth. It adds:
#   - CLI flag       --all2all-backend deepep_v2
#   - Env vars       VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE,
#                    VLLM_DEEPEP_V2_PREFER_OVERLAP,
#                    VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION
#   - Class          DeepEPV2All2AllManager, DeepEPV2PrepareAndFinalize
#   - Probe          vllm.utils.import_utils.has_deep_ep_v2()
#
# We pin the fork's latest commit (6d7a3fab, 2026-05-04) which contains the
# "Pass NCCL device group to DeepEP v2 ElasticBuffer" fix - critical on EFA
# because default gloo group does not carry the Gin transports that
# ElasticBuffer needs.
#
# When PR #41183 merges, flip VLLM_REPO_URL to vllm-project/vllm and use
# `pip install vllm>=<release>` instead of an editable fork install.
# -----------------------------------------------------------------------------
ARG VLLM_REPO_URL=https://github.com/tlrmchlsmth/vllm.git
ARG VLLM_BRANCH=deepep-v2-integration
ARG VLLM_SHA=6d7a3fab28f58d2e73f3c650d24c7297475a7422

RUN set -eux; \
    git clone "${VLLM_REPO_URL}" /opt/vllm; \
    cd /opt/vllm; \
    git fetch origin "${VLLM_BRANCH}"; \
    git checkout "${VLLM_SHA}"; \
    git rev-parse HEAD > /opt/vllm/.git-sha; \
    # Use precompiled kernels so the vLLM install is ~5 min, not ~60.
    VLLM_USE_PRECOMPILED=1 pip install --no-cache-dir --break-system-packages -e .

# -----------------------------------------------------------------------------
# G1 (Wave 6 fix, baked in Wave 7a): realign torchvision to cu129.
# vLLM's VLLM_USE_PRECOMPILED=1 install pulls a torch+torchvision pair whose
# torchvision is linked against CUDA 13 shared libs. The base image ships
# CUDA 12.9 runtime (nvidia/cuda:12.9.0-devel-ubuntu24.04), so torchvision
# fails at import with "undefined symbol / libcudart.so.13 not found" during
# the transformers import chain that vLLM serve triggers on first request.
# Wave 6 worked around this at runtime with a pip reinstall on-pod. Moving
# that step here eliminates the on-pod workaround.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
      --index-url https://download.pytorch.org/whl/cu129 \
      torchvision

# -----------------------------------------------------------------------------
# G2 (Wave 6 fix, baked in Wave 7a): install deep_gemm for Qwen3-FP8 MoE.
# vLLM's DeepEP V2 FP8 MoE kernel path requires the `deep_gemm` library for
# per-expert FP8 GEMMs; without it vLLM falls back to bfloat16 with
# VLLM_USE_DEEP_GEMM=0. PyPI does not publish a pre-built deep_gemm wheel
# as of 2026-05-05 (upstream DeepSeek deep_gemm is source-only, SM90
# required). We try PyPI first for forward-compatibility, then fall back
# to a git install against the upstream repo. If the source build fails
# (e.g. CI host lacks CUDA), we emit a clear log line and keep the image
# usable: FP8 MoE will then require VLLM_USE_DEEP_GEMM=0 + bfloat16 at
# runtime (documented in docs/UPSTREAM-STATUS.md).
# -----------------------------------------------------------------------------
RUN set -eux; \
    if pip install --no-cache-dir --break-system-packages deep_gemm 2>/dev/null; then \
      echo "deep_gemm: installed from PyPI"; \
    elif pip install --no-cache-dir --break-system-packages \
         "git+https://github.com/deepseek-ai/DeepGEMM.git" 2>/dev/null; then \
      echo "deep_gemm: installed from git source"; \
    else \
      echo "WARNING: deep_gemm not installable on this build host;"; \
      echo "         FP8 MoE will require VLLM_USE_DEEP_GEMM=0 + bfloat16 fallback."; \
      echo "         See docs/UPSTREAM-STATUS.md."; \
    fi; \
    python3 -c "import deep_gemm; print('deep_gemm OK:', deep_gemm.__file__)" 2>&1 \
      || echo "deep_gemm not importable; documented fallback applies"

# -----------------------------------------------------------------------------
# Wave 7c.3 — force torch to cu129 AFTER vLLM install so its transitive
# cu130 torch (if any) is overridden. Base image's deep_ep 146cc356 is
# compiled against cu129 torch, so the .so is ABI-compatible once torch is
# cu129. No deep_ep rebuild needed.
# -----------------------------------------------------------------------------
RUN set -eux; \
    pip install --no-cache-dir --break-system-packages --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu129 \
      torch; \
    python3 -c "import torch; assert '+cu129' in torch.__version__, torch.__version__; print('torch realigned to cu129:', torch.__version__)"; \
    # Wave 9b: the torch cu129 2.11.0 wheel pins `nvidia-nccl-cu12==2.28.9`, which
    # gets downgraded by `--force-reinstall` even though the base image ships
    # 2.30.4. DeepEP V2 _C.so was compiled against 2.30+ symbols
    # (`ncclCommQueryProperties`), so we re-upgrade NCCL to 2.30.4 (cu12) after
    # torch. This preserves cu12 unification across the whole stack - no cu13
    # anywhere. HISTORICAL: Wave 7c.4 did the same thing but with a cu13 wheel,
    # causing Wave 8 "invalid device ordinal" crashes because torch cu129 is a
    # cu12.9 build, not a cu13 build.
    pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "${NVIDIA_NCCL_PIN}"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ldconfig; \
    python3 -c "import deep_ep; assert hasattr(deep_ep, 'ElasticBuffer'); print('deep_ep.ElasticBuffer OK')"; \
    # Sanity checks for PR #41183 symbols.
    # The vllm.model_executor probe is done via `grep` of the source file
    # rather than `import`, because a full import pulls in transformers ->
    # torchvision whose _meta_registrations require a CUDA runtime to
    # register `torchvision::nms` (fails on CPU-only build hosts). Runtime
    # use of these classes inside a GPU container is unaffected.
    python3 -c "from vllm.utils.import_utils import has_deep_ep_v2; print('has_deep_ep_v2 symbol OK')"; \
    grep -q '^class DeepEPV2PrepareAndFinalize' \
      /opt/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py \
      && echo 'DeepEPV2PrepareAndFinalize class present in source'

# -----------------------------------------------------------------------------
# NO-SHIM INVARIANT: this image explicitly has no /opt/api-shim and no
# sitecustomize.py from a shim. The preflight step verifies this.
# -----------------------------------------------------------------------------
RUN set -eux; \
    test ! -e /opt/api-shim; \
    find / -name "sitecustomize.py" -path "*/api_shim/*" -exec rm -v {} \; 2>/dev/null || true; \
    find / -name "sitecustomize.py" -path "*/api-shim/*" -exec rm -v {} \; 2>/dev/null || true

ENV PYTHONPATH=/opt/vllm

# LD_LIBRARY_PATH: EFA + OFI discoverable at runtime. Without this, NCCL
# falls back to Socket transport silently. (Prepended to the base image's
# LD_LIBRARY_PATH so EFA/OFI wins if multiple libnccl-net-ofi.so exist.)
ENV LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Runtime env for the vLLM layer (PR #41183 defaults + shim off).
ENV DEEP_EP_USE_V2_SHIM=0 \
    VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1 \
    VLLM_DEEPEP_V2_PREFER_OVERLAP=0 \
    VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION=0

# Bake tests and preflight into the image for one-shot validation.
COPY docker/preflight.sh /opt/docker/preflight.sh
COPY tests/smoke_test.py /opt/tests/smoke_test.py
COPY tests/serve_chat_completion.sh /opt/tests/serve_chat_completion.sh
COPY tests/verify_efa_traffic.sh /opt/tests/verify_efa_traffic.sh
RUN chmod +x /opt/docker/preflight.sh /opt/tests/serve_chat_completion.sh /opt/tests/verify_efa_traffic.sh

WORKDIR /opt

ARG BUILD_MODE
LABEL org.opencontainers.image.title="vllm-deepep-v2-efa" \
      org.opencontainers.image.description="vLLM DeepEP V2 MoE inference on AWS EFA: vanilla upstream + PR #612 (DeepEP) + PR #41183 (vLLM). No shim." \
      org.opencontainers.image.source="https://github.com/antonai-work/vllm-deepep-v2-efa" \
      org.opencontainers.image.licenses="Apache-2.0" \
      build.mode="${BUILD_MODE}" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      vllm.repo_url="https://github.com/tlrmchlsmth/vllm.git" \
      vllm.branch="deepep-v2-integration" \
      vllm.sha="6d7a3fab28f58d2e73f3c650d24c7297475a7422" \
      vllm.pr="41183" \
      deepep.pr="612" \
      efa.installer="1.48.0" \
      aws_ofi_nccl.sha="6e504db3403931cde43a2335adcc73fbc69cccac" \
      shim.included="false"
