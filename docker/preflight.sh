#!/bin/bash
# Preflight for vllm-deepep-v2-efa: seven checks that must ALL pass before
# declaring the image ready for EFA serving.
#
# Exit 0 = all pass. Exit non-zero = fail fast with failing check count.
#
# Check 1: aws-ofi-nccl plugin on ld path so NCCL can load OFI transport
# Check 2: NCCL 2.30.4 runtime (required by DeepEP V2's ncclTeamTagRail)
# Check 3: deep_ep.ElasticBuffer importable (V2 API surface present)
# Check 4: vLLM PR #41183 symbols present (has_deep_ep_v2 probe True)
# Check 5: DeepEPV2PrepareAndFinalize importable (vLLM native V2 backend)
# Check 6: No V1->V2 compat shim directory /opt/api-shim present
# Check 7: DEEP_EP_USE_V2_SHIM env var is 0

set -uo pipefail

HEADER="============================================================"

fail_count=0
pass_count=0

report() {
    local status="$1"
    local label="$2"
    local detail="$3"
    if [[ "${status}" == "PASS" ]]; then
        echo "  [PASS] ${label}"
        [[ -n "${detail}" ]] && echo "         ${detail}"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] ${label}"
        [[ -n "${detail}" ]] && echo "         ${detail}"
        fail_count=$((fail_count + 1))
    fi
}

echo "${HEADER}"
echo "vllm-deepep-v2-efa preflight  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "${HEADER}"

# Check 1: aws-ofi-nccl plugin
echo
echo "Check 1/7: aws-ofi-nccl plugin on ld path"
ldconfig_out="$(ldconfig -p 2>/dev/null | grep -E 'libnccl-net-ofi\.so|libnccl-net\.so' || true)"
echo "  ldconfig -p | grep libnccl-net:"
if [[ -z "${ldconfig_out}" ]]; then
    report "FAIL" "libnccl-net plugin NOT on ld path" \
        "ldconfig sees no libnccl-net-ofi.so -- NCCL will fall back to sockets"
else
    echo "${ldconfig_out}" | sed 's/^/    /'
    report "PASS" "libnccl-net plugin on ld path" \
        "NCCL will discover OFI plugin on AWS EFA"
fi

# Check 2: NCCL runtime version >= 2.30.4
echo
echo "Check 2/7: runtime NCCL version >= 2.30.4"
nccl_ver_out="$(python3 -c "
import ctypes, ctypes.util
# Try loading libnccl.so directly (respects LD_LIBRARY_PATH and pip wheel path).
lib_path = ctypes.util.find_library('nccl')
handles = []
for cand in [lib_path, 'libnccl.so.2', 'libnccl.so']:
    if not cand:
        continue
    try:
        handles.append(ctypes.CDLL(cand))
        break
    except OSError:
        continue
if not handles:
    print('STATUS=fail')
    print('reason=libnccl not loadable via ctypes')
    raise SystemExit(0)
lib = handles[0]
lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.ncclGetVersion.restype = ctypes.c_int
ver = ctypes.c_int(0)
lib.ncclGetVersion(ctypes.byref(ver))
v = ver.value
major = v // 10000
minor = (v // 100) % 100
patch = v % 100
print('runtime NCCL = %d.%d.%d (raw=%d)' % (major, minor, patch, v))
print('STATUS=%s' % ('pass' if v >= 23004 else 'fail'))
" 2>&1)"
echo "${nccl_ver_out}" | sed 's/^/    /'
if echo "${nccl_ver_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "NCCL >= 2.30.4 at runtime" \
        "DeepEP V2's ncclTeamTagRail symbol is available"
else
    report "FAIL" "NCCL < 2.30.4 at runtime" \
        "DeepEP V2 JIT will produce undefined-symbol errors at first dispatch"
fi

# Check 3: deep_ep.ElasticBuffer importable
echo
echo "Check 3/7: deep_ep.ElasticBuffer importable (V2 API surface)"
eb_out="$(python3 -c "
import deep_ep
cls = getattr(deep_ep, 'ElasticBuffer', None)
if cls is None:
    print('STATUS=fail')
    print('reason=deep_ep.ElasticBuffer is None')
else:
    print('deep_ep.ElasticBuffer = %s' % cls)
    print('STATUS=pass')
" 2>&1)"
echo "${eb_out}" | sed 's/^/    /'
if echo "${eb_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "deep_ep.ElasticBuffer importable" \
        "V2 API surface present"
else
    report "FAIL" "deep_ep.ElasticBuffer missing" \
        "DeepEP install is V1-only or broken"
fi

# Check 4: vLLM has_deep_ep_v2 probe True
echo
echo "Check 4/7: vllm.utils.import_utils.has_deep_ep_v2() = True (PR #41183)"
hasv2_out="$(python3 -c "
from vllm.utils.import_utils import has_deep_ep_v2
val = has_deep_ep_v2()
print('has_deep_ep_v2() = %s' % val)
print('STATUS=%s' % ('pass' if val else 'fail'))
" 2>&1)"
echo "${hasv2_out}" | sed 's/^/    /'
if echo "${hasv2_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "has_deep_ep_v2() returns True" \
        "vLLM PR #41183 probe is active"
else
    report "FAIL" "has_deep_ep_v2() returns False" \
        "Either PR #41183 not applied, NCCL < 2.30.4, or deep_ep missing"
fi

# Check 5: DeepEPV2PrepareAndFinalize importable
echo
echo "Check 5/7: DeepEPV2PrepareAndFinalize importable (PR #41183 class)"
pf_out="$(python3 -c "
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import DeepEPV2PrepareAndFinalize
print('DeepEPV2PrepareAndFinalize = %s' % DeepEPV2PrepareAndFinalize)
print('STATUS=pass')
" 2>&1)"
echo "${pf_out}" | sed 's/^/    /'
if echo "${pf_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "DeepEPV2PrepareAndFinalize importable" \
        "PR #41183 code path wired into vLLM"
else
    report "FAIL" "DeepEPV2PrepareAndFinalize missing" \
        "vLLM fork SHA does not contain PR #41183 changes"
fi

# Check 6: /opt/api-shim not present
echo
echo "Check 6/7: /opt/api-shim must not exist (no V1->V2 compat shim)"
if [[ -e /opt/api-shim ]]; then
    report "FAIL" "/opt/api-shim exists" "$(ls -la /opt/api-shim 2>&1 | head -3)"
else
    report "PASS" "/opt/api-shim absent" "no shim installed"
fi

# Check 7: DEEP_EP_USE_V2_SHIM=0
echo
echo "Check 7/7: DEEP_EP_USE_V2_SHIM env var must be 0"
shim_env="${DEEP_EP_USE_V2_SHIM:-<unset>}"
if [[ "${shim_env}" == "0" ]]; then
    report "PASS" "DEEP_EP_USE_V2_SHIM=0" "shim disabled by env"
else
    report "FAIL" "DEEP_EP_USE_V2_SHIM=${shim_env}" "must be 0 to prove native V2"
fi

echo
echo "${HEADER}"
echo "Preflight result: ${pass_count} PASS, ${fail_count} FAIL"
echo "${HEADER}"

exit "${fail_count}"
