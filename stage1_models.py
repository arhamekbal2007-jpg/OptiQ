"""Stage 1 models.

Quantum / hybrid model : QuantumKernelLR
    classical preprocessing -> classical dimensionality reduction to n_qubits features
    -> ZZ-feature-map quantum state per customer (statevector simulation)
    -> quantum kernel K(x, x') = |<psi(x)|psi(x')>|^2
    -> classical L2 logistic regression on the kernel row ("empirical kernel map")
    The quantum part is the kernel (the similarity measure); the probability model is classical.

Classical reference    : logistic regression (+ a prior-only 'base rate' reference).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from sim import apply_all_1q, bit_table, H, zero_state


def zz_feature_states(angles: np.ndarray, reps: int = 2) -> np.ndarray:
    """Statevectors of a Qiskit-style ZZFeatureMap (linear entanglement replaced by full pairs).

    Per repetition:  H on every qubit; P(2 x_i) on every qubit;
                     for every pair i<j: CX(i,j) P_j(2 (pi-x_i)(pi-x_j)) CX(i,j)
    The phase part is diagonal, so it is applied as one exp(i*phase) multiplication.
    """
    N, n = angles.shape
    bits = bit_table(n).astype(float)  # (2^n, n)
    phase = 2.0 * angles @ bits.T  # (N, 2^n)
    for i in range(n):
        for j in range(i + 1, n):
            xor = np.abs(bits[:, i] - bits[:, j])
            phase += 2.0 * ((np.pi - angles[:, i]) * (np.pi - angles[:, j]))[:, None] * xor[None, :]
    D = np.exp(1j * phase)
    psi = zero_state(n, N)
    for _ in range(reps):
        psi = apply_all_1q(psi, H, n)
        psi = psi * D
    return psi


def fidelity_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A @ B.conj().T) ** 2


class QuantumKernelLR(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=4, reducer="pca", scale=1.0, reps=2, C=1.0, random_state=42):
        self.n_qubits = n_qubits
        self.reducer = reducer
        self.scale = scale
        self.reps = reps
        self.C = C
        self.random_state = random_state

    def _angles(self, Xr):
        Z = np.clip(self.mm_.transform(Xr), 0.0, 1.0)
        return Z * np.pi * self.scale

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.array([0, 1])
        if self.reducer == "pca":
            self.red_ = PCA(n_components=self.n_qubits, random_state=self.random_state).fit(X)
        else:
            self.red_ = SelectKBest(f_classif, k=self.n_qubits).fit(X, y)
        Xr = self.red_.transform(X)
        self.mm_ = MinMaxScaler().fit(Xr)
        self.train_states_ = zz_feature_states(self._angles(Xr), self.reps)
        K = fidelity_kernel(self.train_states_, self.train_states_)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state).fit(K, y)
        return self

    def predict_proba(self, X):
        Xr = self.red_.transform(np.asarray(X, float))
        states = zz_feature_states(self._angles(Xr), self.reps)
        return self.lr_.predict_proba(fidelity_kernel(states, self.train_states_))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
