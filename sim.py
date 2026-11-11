"""Minimal statevector / density-matrix simulator written in NumPy.

Conventions (identical to Qiskit):
  * basis index = sum_q b_q * 2**q, i.e. qubit 0 is the LEAST significant bit
  * a bitstring is printed with qubit 0 on the RIGHT ("q_{n-1} ... q_1 q_0")

This is used because the execution environment for development had no quantum SDK installed.
The Qiskit ports in ../qiskit_port build the same circuits and should reproduce these numbers.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def zero_state(n, batch=None):
    shape = (2**n,) if batch is None else (batch, 2**n)
    s = np.zeros(shape, dtype=complex)
    s[..., 0] = 1.0
    return s


def apply_1q(state, U, q, n):
    """Apply a single-qubit gate U to qubit q of `state` (shape (..., 2**n))."""
    lead = state.shape[:-1]
    s = state.reshape(lead + (2 ** (n - q - 1), 2, 2**q))
    s = np.einsum("ab,...xby->...xay", U, s)
    return s.reshape(lead + (2**n,))


def apply_all_1q(state, U, n):
    for q in range(n):
        state = apply_1q(state, U, q, n)
    return state


def bit_table(n):
    """bits[i, q] = value of qubit q in basis state i (shape (2**n, n))."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def probabilities(state):
    return np.abs(state) ** 2


def bitstring(i, n):
    return format(int(i), f"0{n}b")  # q_{n-1} ... q_0


def sample_counts(probs, shots, rng):
    """Multinomial sampling of measurement outcomes; returns {bitstring: count}."""
    n = int(np.log2(len(probs)))
    probs = probs / probs.sum()
    draws = rng.multinomial(shots, probs)
    return {bitstring(i, n): int(c) for i, c in enumerate(draws) if c > 0}


# ------------------------- density-matrix helpers (small n only) -------------------------
def dm_from_state(psi):
    return np.outer(psi, psi.conj())


def dm_apply_1q(rho, U, q, n):
    full = _embed_1q(U, q, n)
    return full @ rho @ full.conj().T


def _embed_1q(U, q, n):
    ops = [I2] * n
    ops[q] = U
    full = np.array([[1.0 + 0j]])
    for q_ in range(n - 1, -1, -1):  # most significant qubit first in the Kronecker product
        full = np.kron(full, ops[q_])
    return full


def dm_depolarize_1q(rho, p, q, n):
    """Single-qubit depolarizing channel: rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)."""
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    out = (1 - p) * rho
    for P in (X, Y, Z):
        full = _embed_1q(P, q, n)
        out = out + (p / 3) * (full @ rho @ full.conj().T)
    return out
