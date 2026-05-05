# syntax=docker/dockerfile:1.10
# =============================================================================
# vllm-deepep-v2-efa - vLLM MoE inference on DeepEP V2 over AWS EFA
# =============================================================================
#
# One self-contained Dockerfile that layers the two open upstream PRs this
# stack needs on top of pure-upstream substrate. Builds from the vanilla
# public NVIDIA CUDA base image; no pre-built base image required.
#
# Pinned layers:
#   Base OS + CUDA       nvidia/cuda:12.9.0-devel-ubuntu24.04   vanilla
#   EFA userspace        aws-efa-installer 1.48.0               vanilla (--build-ngc)
#   aws-ofi-nccl         6e504db3403931cde43a2335adcc73fbc69cccac  vanilla
#                        ("gin: Size active_put_signal to full sequence
#                         number space", upstream fix 2026-04-24)
#   NCCL                 nvidia-nccl-cu13 >= 2.30.4             vanilla
#   GDRCopy              v2.5.1                                 vanilla
#   DeepEP V2            deepseek-ai/DeepEP@b306af0 + patches
#                        0001-0003 (PR #612)                    OUR PATCHES
#   vLLM                 github.com/tlrmchlsmth/vllm @           vLLM PR #41183
#                        deepep-v2-integration                   HEAD (pinned SHA)
#                        = vanilla vllm-project/vllm @ <main-base> + PR #41183
#   api-shim             NOT INCLUDED                           by design
#                        DEEP_EP_USE_V2_SHIM=0 at runtime; vLLM calls
#                        deep_ep.ElasticBuffer natively via PR #41183.
#
# Build FROM REPO ROOT:
#   docker build -f docker/Dockerfile -t <your-registry>/<repo>:<tag> .
# or equivalently:
#   docker/build.sh <your-registry>/<repo>:<tag>
#
# Preflight gate (inside running container):
#   docker run --rm <image> bash /opt/docker/preflight.sh
#   -> expected: "7/7 checks PASS"
#
# Runtime gate (2-pod EFA cluster):
#   kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
#   -> expected: /v1/chat/completions returns a real JSON response
# =============================================================================

FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# System deps (build toolchain + openmpi + numa/libnl for rdma-core)
# Intentionally NOT installing rdma-core/libibverbs-dev from apt -- the EFA
# installer below lands matched v61 .debs; apt v50 would block them.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget tar xz-utils file \
    autoconf automake libtool pkg-config cmake ninja-build patchelf \
    python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev openmpi-bin \
    libnuma-dev libnl-3-dev libnl-route-3-dev \
    kmod pciutils jq \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# EFA userspace (aws-efa-installer 1.48.0, --build-ngc path)
# Manually dpkg -i --auto-deconfigure to upgrade ibverbs-providers v50 -> v61
# since libfabric1-aws 2.4.0amzn3.0 declares Depends: ibverbs-providers >= 59.
# Runtime ABI is stable across these versions; it's a packaging-metadata gap.
# -----------------------------------------------------------------------------
ARG EFA_INSTALLER_VERSION=1.48.0
RUN set -eux; \
    cd /tmp; \
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
    rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer.tar.gz

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64

# -----------------------------------------------------------------------------
# GDRCopy 2.5.1 userspace (host ships the kernel module)
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --depth=1 --branch v2.5.1 \
      https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy; \
    cd /tmp/gdrcopy; \
    CUDA=/usr/local/cuda make lib_install; \
    ldconfig; \
    rm -rf /tmp/gdrcopy

# -----------------------------------------------------------------------------
# PyTorch + NCCL 2.30.4 via pip (DeepEP V2 JIT finds NCCL headers here)
#
# vLLM PR #41183 explicitly checks the *runtime* NCCL version via
# ncclGetVersion(). PyTorch pins nvidia-nccl-cu12==2.27.5 in its cu128
# metadata; DeepEP V2's _C.so is linked against 2.30.4 symbols
# (ncclTeamTagRail, etc.). We need a matched 2.30.4 at runtime.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu129 \
 || pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128

# numpy<2 required by torch.distributed.all_gather_object (used by V2 init).
RUN pip install --no-cache-dir --break-system-packages "numpy<2"

# Purge apt-installed libnccl2 that would shadow the 2.30.4 pip wheel.
RUN set -eux; \
    apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true; \
    apt-mark hold libnccl2 libnccl-dev 2>/dev/null || true; \
    rm -f /lib/x86_64-linux-gnu/libnccl.so* /usr/lib/x86_64-linux-gnu/libnccl.so*; \
    ldconfig

# Install NCCL 2.30.4 via the cu13 pip wheel (matching DeepEP V2 symbol
# requirements). Also drop the cu12 pin that torch may have pulled in.
RUN set -eux; \
    pip uninstall -y --break-system-packages nvidia-nccl-cu12 || true; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      "nvidia-nccl-cu13>=2.30.4"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    ldconfig

# NVSHMEM wheel - DeepEP V2's setup.py still links against libnvshmem_host.so
# even when NCCL Gin is the primary path (upstream TODO). We do not use
# NVSHMEM transports at runtime; NCCL_GIN_TYPE=2 keeps everything on Gin.
RUN pip install --no-cache-dir --break-system-packages \
      "nvidia-nvshmem-cu12>=3.3.9" \
 && NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)" \
 && test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}" \
 && HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)" \
 && ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so" \
 && echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf \
 && ldconfig

# -----------------------------------------------------------------------------
# aws-ofi-nccl built from source at 6e504db (2026-04-24):
#   "gin: Size active_put_signal to full sequence number space"
# Upstream fix for the 128-slot Gin ring overflow. Must build AFTER
# nvidia-nccl-cu13 so --with-nccl finds matching 2.30.4 headers.
# -----------------------------------------------------------------------------
ARG AWS_OFI_NCCL_SHA=6e504db3403931cde43a2335adcc73fbc69cccac
RUN set -eux; \
    git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl; \
    cd /tmp/aws-ofi-nccl; \
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
    rm -rf /tmp/aws-ofi-nccl

# -----------------------------------------------------------------------------
# DeepEP V2 with patches 0001-0003 (PR #612)
#
# These three patches:
#   0001: cap auto-QP at 2 on EFA to avoid 128-slot GIN ring overflow
#   0002: add EFA fast path in get_rdma_gbs() so SM auto-sizing is sane
#   0003: raise dispatch kScaleoutUpdateInterval from 3 to 16 (EFA proxy tuning)
#
# Base: deepseek-ai/DeepEP@b306af0 (main at merge of V2 release PR #605).
# When PR #612 merges upstream, flip DEEPEP_REPO_URL back to vanilla and
# drop the patch apply step.
# -----------------------------------------------------------------------------
ARG DEEPEP_REPO_URL=https://github.com/deepseek-ai/DeepEP.git
ARG DEEPEP_BASE_SHA=b306af06af0d8d62b4b57a7e9664afa8c8f9d40e

COPY patches/0001-aws-efa-cap-auto-QP-at-2-on-EFA-to-avoid-128-slot-GI.patch /tmp/patches/0001-deepep.patch
COPY patches/0002-aws-efa-add-EFA-fast-path-in-get_rdma_gbs-to-fix-SM-.patch  /tmp/patches/0002-deepep.patch
COPY patches/0003-aws-efa-raise-dispatch-kScaleoutUpdateInterval-from-.patch  /tmp/patches/0003-deepep.patch

RUN set -eux; \
    git config --global user.email "build@vllm-deepep-v2-efa.local"; \
    git config --global user.name "build"; \
    git clone --recurse-submodules "${DEEPEP_REPO_URL}" /opt/DeepEP; \
    cd /opt/DeepEP; \
    git checkout "${DEEPEP_BASE_SHA}"; \
    git am /tmp/patches/0001-deepep.patch \
           /tmp/patches/0002-deepep.patch \
           /tmp/patches/0003-deepep.patch; \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"; \
    rm -rf /tmp/patches

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

# PyTorch's torch wheel pulled in nvidia-nccl-cu12 as a dep and may have
# clobbered the cu13 2.30.4 symlink. Restore it.
RUN set -eux; \
    pip uninstall -y --break-system-packages nvidia-nccl-cu12 || true; \
    pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13>=2.30.4"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ldconfig; \
    # Also recompile DeepEP V2 against torch's ABI so _C.so matches.
    cd /opt/DeepEP; \
    export LIBRARY_PATH="${NCCL_LIB}:${LIBRARY_PATH:-}"; \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    pip install --no-build-isolation --break-system-packages --force-reinstall --no-deps -e .; \
    # Sanity checks for PR #41183 symbols.
    python3 -c "from vllm.utils.import_utils import has_deep_ep_v2; print('has_deep_ep_v2 symbol OK')"; \
    python3 -c "from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import DeepEPV2PrepareAndFinalize; print('DeepEPV2PrepareAndFinalize OK')"; \
    python3 -c "import deep_ep; assert hasattr(deep_ep, 'ElasticBuffer'); print('deep_ep.ElasticBuffer OK')"

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
# falls back to Socket transport silently.
ENV LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Runtime env (matches the validated cluster config and PR #41183 defaults).
ENV DEEP_EP_USE_V2_SHIM=0 \
    NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_CUMEM_HOST_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_IGNORE_DISABLED_P2P=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0 \
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

LABEL org.opencontainers.image.title="vllm-deepep-v2-efa" \
      org.opencontainers.image.description="vLLM DeepEP V2 MoE inference on AWS EFA: vanilla upstream + PR #612 (DeepEP) + PR #41183 (vLLM). No shim." \
      org.opencontainers.image.source="https://github.com/antonai-work/vllm-deepep-v2-efa" \
      org.opencontainers.image.licenses="Apache-2.0" \
      deepep.base_sha="${DEEPEP_BASE_SHA}" \
      vllm.repo_url="${VLLM_REPO_URL}" \
      vllm.branch="${VLLM_BRANCH}" \
      vllm.sha="${VLLM_SHA}" \
      vllm.pr="41183" \
      deepep.pr="612" \
      efa.installer="${EFA_INSTALLER_VERSION}" \
      aws_ofi_nccl.sha="${AWS_OFI_NCCL_SHA}" \
      shim.included="false"
