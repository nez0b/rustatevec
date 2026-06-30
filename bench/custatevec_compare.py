#!/usr/bin/env python3
"""Apples-to-apples QFT(n) comparison: NVIDIA cuStateVec vs qsv's CudaBackend, on the same L40S.

Both evolve QFT(n) from |0...0> with the SAME gate sequence (H, controlled-phase, bit-reversal
SWAPs) at f64 precision, timing only state evolution with a single device sync at the end (qsv's
batched `execute` does likewise). qsv timing comes from the `qft_time` example (shelled out).

Run inside the venv:
  .venv/bin/python bench/custatevec_compare.py
"""
import math, os, subprocess, sys, time

import numpy as np
import cupy as cp
from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

C64 = cudaDataType.CUDA_C_64F
COMP = ComputeType.COMPUTE_64F
ROW = cusv.MatrixLayout.ROW

H = (np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)).copy(order="C")
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128).copy(order="C")


def pphase(theta):
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128).copy(order="C")


def apply(handle, sv, n, mat, targets, controls=()):
    ws_size = cusv.apply_matrix_get_workspace_size(
        handle, C64, n, mat.ctypes.data, C64, ROW, 0, len(targets), len(controls), COMP)
    ws = cp.empty(ws_size, dtype=cp.uint8) if ws_size > 0 else None
    ws_ptr = ws.data.ptr if ws is not None else 0
    cusv.apply_matrix(
        handle, sv.data.ptr, C64, n,
        mat.ctypes.data, C64, ROW, 0,
        list(targets), len(targets),
        list(controls), [1] * len(controls), len(controls),
        COMP, ws_ptr, ws_size)


def qft_custatevec(handle, n):
    sv = cp.zeros(1 << n, dtype=cp.complex128)
    sv[0] = 1.0
    for j in range(n):
        apply(handle, sv, n, H, [j])
        for k in range(j + 1, n):
            apply(handle, sv, n, pphase(math.pi / (1 << (k - j))), [j], controls=[k])
    for i in range(n // 2):
        apply(handle, sv, n, SWAP, [i, n - 1 - i])
    return sv


def n_gates(n):
    return n + n * (n - 1) // 2 + n // 2  # H + cphases + swaps (matches qsv::circuits::qft)


def time_custatevec(n, reps):
    handle = cusv.create()
    sv = qft_custatevec(handle, n)  # warmup
    cp.cuda.runtime.deviceSynchronize()
    # correctness: QFT|0> is the uniform superposition, every |amp|^2 = 1/2^n
    p0 = float(abs(sv[0]) ** 2)
    assert abs(p0 - 1.0 / (1 << n)) < 1e-9, f"cuStateVec QFT wrong: p0={p0}"
    t = time.perf_counter()
    for _ in range(reps):
        qft_custatevec(handle, n)
    cp.cuda.runtime.deviceSynchronize()
    ms = (time.perf_counter() - t) * 1e3 / reps
    cusv.destroy(handle)
    return ms


def time_qsv(n, reps):
    exe = "target/release/examples/qft_time"
    out = subprocess.run([exe, str(n), str(reps)], capture_output=True, text=True,
                         env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"})
    # parse "qsv n=.. gates=.. ms=X gelem_s=Y"
    for tok in out.stdout.split():
        if tok.startswith("ms="):
            return float(tok[3:])
    raise RuntimeError(f"qsv timing failed: {out.stdout}\n{out.stderr}")


def main():
    ns = [int(x) for x in sys.argv[1:]] or [18, 20, 22, 24, 26]
    print("QFT(n) GPU evolution — cuStateVec vs qsv (L40S, f64, ms; lower is better)")
    print(f"{'n':>3} {'gates':>6} {'cuStateVec':>11} {'qsv':>9} {'qsv/cusv':>9}")
    for n in ns:
        reps = 10 if n <= 22 else 3
        cusv_ms = time_custatevec(n, reps)
        qsv_ms = time_qsv(n, reps)
        print(f"{n:>3} {n_gates(n):>6} {cusv_ms:>11.3f} {qsv_ms:>9.3f} {qsv_ms / cusv_ms:>8.2f}x")


if __name__ == "__main__":
    main()
