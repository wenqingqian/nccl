"""Shared utilities for mask all-reduce tests.

This module loads:
  - a fixed no-mask baseline library
  - multiple mask library versions from /workspace/nccl/libs/v*/

Directory layout:
  /workspace/nccl/libs/
    nomask/libnccl-nomask.so          # fixed baseline, never updated
    v0/libnccl-mask.so                # mask version 0
    v1/libnccl-mask.so                # mask version 1
    ...

The no-mask baseline measures the original NCCL implementation, while each
mask version can be compared against the baseline and against previous versions.
"""

import ctypes
import glob
import os
import re

import torch
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Discover and load NCCL libraries
# -----------------------------------------------------------------------------
NCCL_LIBS_ROOT = "/workspace/nccl/libs"
NCCL_NOMASK_LIB = f"{NCCL_LIBS_ROOT}/nomask/libnccl-nomask.so"


def _discover_mask_versions():
    """Return sorted list of mask versions (e.g. ['v0', 'v1'])."""
    pattern = f"{NCCL_LIBS_ROOT}/v*/libnccl-mask.so"
    versions = []
    for path in glob.glob(pattern):
        dirname = os.path.basename(os.path.dirname(path))
        if re.fullmatch(r"v\d+", dirname):
            versions.append(dirname)
    return sorted(versions, key=lambda s: int(s[1:]))


MASK_VERSIONS = _discover_mask_versions()


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


ncclComm_t = ctypes.c_void_p
NCCL_FLOAT32 = 7
NCCL_SUM = 0


def _setup_nccl_api(lib):
    """Configure argtypes/restype for a loaded NCCL library."""
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
    lib.ncclGetUniqueId.restype = ctypes.c_int
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int


# Load fixed no-mask baseline.
nccl_nomask = ctypes.CDLL(NCCL_NOMASK_LIB)
_setup_nccl_api(nccl_nomask)
nccl_nomask.ncclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
    ncclComm_t, ctypes.c_void_p,
]
nccl_nomask.ncclAllReduce.restype = ctypes.c_int

# Load all mask versions.
_nccl_mask_libs = {}
for ver in MASK_VERSIONS:
    path = f"{NCCL_LIBS_ROOT}/{ver}/libnccl-mask.so"
    lib = ctypes.CDLL(path)
    _setup_nccl_api(lib)
    lib.ncclAllReduce.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ncclComm_t, ctypes.c_void_p,
    ]
    lib.ncclAllReduce.restype = ctypes.c_int
    _nccl_mask_libs[ver] = lib


_comms = {}


def list_mask_versions():
    """Return the discovered mask versions."""
    return list(MASK_VERSIONS)


def _broadcast_uid(rank):
    """Broadcast a 128-byte UID from rank 0 using torch distributed."""
    if rank == 0:
        uid = ncclUniqueId()
        # Use the no-mask lib to generate the UID; the raw bytes are library-agnostic.
        ret = nccl_nomask.ncclGetUniqueId(ctypes.byref(uid))
        assert ret == 0, f"ncclGetUniqueId failed: {ret}"
        uid_bytes = bytes(ctypes.cast(
            ctypes.addressof(uid), ctypes.POINTER(ctypes.c_char * 128)
        ).contents)
    else:
        uid_bytes = b""

    uid_list = [uid_bytes]
    dist.broadcast_object_list(uid_list, src=0)
    uid_bytes = uid_list[0]

    uid = ncclUniqueId()
    ctypes.memmove(ctypes.addressof(uid), uid_bytes, 128)
    return uid


def init_nccl_comm(version="nomask"):
    """Initialize per-rank NCCL comm for the requested library version.

    Args:
        version: "nomask" for the fixed baseline, or "v0"/"v1"/... for a mask version.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    key = version
    if key in _comms:
        return _comms[key]

    uid = _broadcast_uid(rank)
    if version == "nomask":
        nccl_lib = nccl_nomask
    else:
        if version not in _nccl_mask_libs:
            raise ValueError(f"Unknown mask version: {version}. Available: {list(_nccl_mask_libs.keys())}")
        nccl_lib = _nccl_mask_libs[version]

    comm = ncclComm_t()
    ret = nccl_lib.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank)
    assert ret == 0, f"ncclCommInitRank failed for {version}: {ret}"
    _comms[key] = comm
    return comm


def mask_all_reduce(tensor, mask=None, version="v0"):
    """In-place mask all-reduce using the specified mask library version."""
    if not tensor.is_contiguous():
        raise RuntimeError("Tensor must be contiguous")

    if version not in _nccl_mask_libs:
        raise ValueError(f"Unknown mask version: {version}. Available: {list(_nccl_mask_libs.keys())}")

    comm = init_nccl_comm(version)
    stream = torch.cuda.current_stream(tensor.device)
    nccl_lib = _nccl_mask_libs[version]

    extra_ptr = ctypes.c_void_p(mask.data_ptr()) if mask is not None else ctypes.c_void_p(0)
    ret = nccl_lib.ncclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(tensor.data_ptr()),
        extra_ptr,
        tensor.numel(),
        NCCL_FLOAT32,
        NCCL_SUM,
        comm,
        ctypes.c_void_p(stream.cuda_stream),
    )
    assert ret == 0, f"ncclAllReduce (mask {version}) failed: {ret}"
    return tensor


def vanilla_all_reduce(tensor):
    """In-place vanilla all-reduce using the fixed upstream NCCL baseline."""
    if not tensor.is_contiguous():
        raise RuntimeError("Tensor must be contiguous")

    comm = init_nccl_comm("nomask")
    stream = torch.cuda.current_stream(tensor.device)

    ret = nccl_nomask.ncclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(tensor.data_ptr()),
        tensor.numel(),
        NCCL_FLOAT32,
        NCCL_SUM,
        comm,
        ctypes.c_void_p(stream.cuda_stream),
    )
    assert ret == 0, f"ncclAllReduce (nomask) failed: {ret}"
    return tensor


# -----------------------------------------------------------------------------
# Mask generation
# -----------------------------------------------------------------------------
BYTES_PER_PACK = 16  # fp32 pack size observed in NCCL Simple protocol
FLOATS_PER_PACK = BYTES_PER_PACK // 4


def real_sparsity(mask):
    """Compute the effective zero ratio at NCCL pack granularity.

    A pack is considered zero only if all floats in the pack are exactly 0.
    """
    if mask.numel() % FLOATS_PER_PACK != 0:
        # Pad to pack boundary for analysis.
        pad = FLOATS_PER_PACK - (mask.numel() % FLOATS_PER_PACK)
        mask = torch.nn.functional.pad(mask, (0, pad))
    packs = mask.view(-1, FLOATS_PER_PACK)
    zero_packs = (packs == 0.0).all(dim=1).sum().item()
    return zero_packs / packs.size(0)


def generate_mask(size, target_sparsity, pattern="block", device="cuda"):
    """Generate a pack-aligned fp32 mask with the requested target sparsity.

    Args:
        size: number of fp32 elements.  Will be rounded up to pack boundary.
        target_sparsity: desired ratio of zero packs (0.0 - 1.0).
        pattern: one of
            - "block": large contiguous blocks of zeros/ones (e.g. 00001111).
            - "alternating": pack-level alternating zeros/ones (0101).
            - "random": random pack-level distribution.
            - "mixed": intermediate between block and alternating.
    Returns:
        mask: fp32 tensor on `device`.
        actual_sparsity: effective zero pack ratio.
    """
    n_packs = (size + FLOATS_PER_PACK - 1) // FLOATS_PER_PACK
    n_zeros = int(round(n_packs * target_sparsity))
    n_ones = n_packs - n_zeros

    if pattern == "block":
        pack_mask = torch.cat([
            torch.zeros(n_zeros, dtype=torch.float32),
            torch.ones(n_ones, dtype=torch.float32),
        ])
    elif pattern == "alternating":
        # Pack-level 0101 pattern with the target ratio.
        # Zero packs are placed every `step` positions, so zero ratio = 1/step.
        pack_mask = torch.zeros(n_packs, dtype=torch.float32)
        step = max(2, int(round(1.0 / (target_sparsity + 1e-9))))
        pack_mask[torch.arange(n_packs) % step != 0] = 1.0
    elif pattern == "random":
        generator = torch.Generator().manual_seed(42 + int(target_sparsity * 100))
        perm = torch.randperm(n_packs, generator=generator)
        pack_mask = torch.zeros(n_packs, dtype=torch.float32)
        pack_mask[perm[:n_ones]] = 1.0
    elif pattern == "mixed":
        # Blocks of 2 packs: helps interpolate between block (large) and
        # alternating (single pack).
        block = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        repeats = (n_packs + 3) // 4
        pack_mask = block.repeat(repeats)[:n_packs]
        # Adjust to target sparsity by flipping random packs.
        actual = (pack_mask == 0.0).sum().item() / n_packs
        diff = n_zeros - int(round(actual * n_packs))
        if diff > 0:
            ones = (pack_mask == 1.0).nonzero(as_tuple=True)[0]
            idx = ones[torch.randperm(ones.numel(), generator=torch.Generator().manual_seed(42))[:diff]]
            pack_mask[idx] = 0.0
        elif diff < 0:
            zeros = (pack_mask == 0.0).nonzero(as_tuple=True)[0]
            idx = zeros[torch.randperm(zeros.numel(), generator=torch.Generator().manual_seed(42))[:abs(diff)]]
            pack_mask[idx] = 1.0
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    mask = pack_mask.repeat_interleave(FLOATS_PER_PACK)[:size].to(device)
    return mask, real_sparsity(mask)


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------
def timed_all_reduce(tensor, mask=None, version="v0", warmup=5, iters=10):
    """Run warmup + timed iterations and return average latency in ms.

    Args:
        tensor: input tensor.
        mask: mask tensor, or None to use the no-mask path.  When None and
            version is "nomask", the fixed upstream NCCL baseline is used;
            otherwise the requested mask library version is used with a NULL
            mask pointer.
        version: mask version to use; "nomask" selects the fixed baseline.
        warmup: number of warmup iterations.
        iters: number of timed iterations.
    """
    stream = torch.cuda.current_stream(tensor.device)
    for _ in range(warmup):
        if mask is None:
            if version == "nomask":
                vanilla_all_reduce(tensor.clone())
            else:
                mask_all_reduce(tensor.clone(), None, version=version)
        else:
            mask_all_reduce(tensor.clone(), mask, version=version)
    stream.synchronize()

    times = []
    for _ in range(iters):
        # Prepare the input before the timed region so clone cost is excluded.
        t = tensor.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if mask is None:
            if version == "nomask":
                vanilla_all_reduce(t)
            else:
                mask_all_reduce(t, None, version=version)
        else:
            mask_all_reduce(t, mask, version=version)
        end.record(stream)
        stream.synchronize()
        times.append(start.elapsed_time(end))

    # Drop the fastest 1/3 and slowest 1/3, then average the remaining middle.
    times.sort()
    drop = iters // 3
    trimmed = times[drop:iters - drop]
    return sum(trimmed) / len(trimmed)


def free_memory():
    """Delete references and clear CUDA cache."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
