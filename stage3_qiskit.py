"""Qiskit + Aer port of Stage 3 (needs: pip install qiskit qiskit-aer).

IMPORTANT: this script was written alongside the NumPy implementation in ../src/stage3.py but was NOT executed
in the environment where the results were produced (no quantum SDK was installable there). Run it on your own
machine / Colab; it should agree with ../stage3_results.json up to shot noise. It also prints the transpiled
circuit statistics (depth, 1q/2q gate counts, SWAP count) that the submission guidelines ask for.
"""
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

ROOT = Path(__file__).resolve().parents[1]
ref = json.loads((ROOT / "stage3_results.json").read_text())
p = np.array(ref["stage1_probabilities"])          # qubit i <-> ref["selected_ids"][i]
ids = ref["selected_ids"]
n, shots, K = len(p), ref["shots"], 3
noise = ref["noise_model_assumed"]

# ---- circuit: RY(2 asin sqrt(p_i)) on qubit i, measure all ----
qc = QuantumCircuit(n, n)
for i, pi in enumerate(p):
    qc.ry(2 * np.arcsin(np.sqrt(pi)), i)
qc.measure(range(n), range(n))
print(qc.draw())


def success_prob(counts):
    # Qiskit bitstrings print q_{n-1} ... q_0 (qubit 0 rightmost); only the NUMBER of ones matters here
    return sum(c for bs, c in counts.items() if bs.count("1") >= K) / sum(counts.values())


# ---- ideal, finite shots ----
ideal = AerSimulator(seed_simulator=2026)
counts = ideal.run(transpile(qc, ideal), shots=shots).result().get_counts()
print("ideal finite-shot estimate :", success_prob(counts), "| exact:", ref["exact_probability"])

# per-customer marginals (qubit i is character n-1-i of the printed string)
for i, cid in enumerate(ids):
    m = sum(c for bs, c in counts.items() if bs[n - 1 - i] == "1") / shots
    print(f"  {cid} (q{i}): measured P(1) = {m:.4f}   Stage-1 p = {p[i]:.4f}")

# ---- noisy (same ILLUSTRATIVE noise as the NumPy run: 1q depolarizing on RY + asymmetric readout error) ----
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(noise["depolarizing_1q"], 1), ["ry"])
p01, p10 = noise["readout_p01"], noise["readout_p10"]
nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))  # rows: true 0/1, cols: measured 0/1
noisy = AerSimulator(noise_model=nm, seed_simulator=2027)
counts_n = noisy.run(transpile(qc, noisy, optimization_level=0), shots=shots).result().get_counts()
print("noisy (unmitigated) estimate:", success_prob(counts_n))

# ---- readout mitigation: invert the tensor-product confusion matrix (little-endian => kron from q_{n-1} to q_0) ----
C1 = np.array([[1 - p01, p10], [p01, 1 - p10]])  # rows: measured, cols: true
C = np.array([[1.0]])
for _ in range(n):
    C = np.kron(C, C1)
emp = np.zeros(2**n)
for bs, c in counts_n.items():
    emp[int(bs, 2)] = c / shots
quasi = np.linalg.solve(C, emp)
popcount = np.array([bin(i).count("1") for i in range(2**n)])
readout_mit_est = float(quasi[popcount >= K].sum())
print("noisy + readout-mitigated   :", readout_mit_est)

# ---- Zero-Noise Extrapolation (ZNE) via Unitary Gate Folding & Layered Mitigation ----
# We fold RY(theta): RY(theta) * (RY(-theta) * RY(theta))^m gives scale factors lambda = 1 + 2m (1, 3, 5).
# Layered mitigation: apply readout matrix inversion at each noise level, then extrapolate gate noise to lambda -> 0.
def build_folded_circuit(scale_factor_m):
    qc_f = QuantumCircuit(n, n)
    for i, pi in enumerate(p):
        theta = 2 * np.arcsin(np.sqrt(pi))
        qc_f.ry(theta, i)
        for _ in range(scale_factor_m):
            qc_f.ry(-theta, i)
            qc_f.ry(theta, i)
    qc_f.measure(range(n), range(n))
    return qc_f

scale_factors = [1, 3, 5]
zne_raw = []
zne_readout_mit = []
for m, lam in enumerate(scale_factors):
    qc_m = build_folded_circuit(m)
    c_m = noisy.run(transpile(qc_m, noisy, optimization_level=0), shots=shots).result().get_counts()
    p_raw = success_prob(c_m)
    zne_raw.append(p_raw)
    
    # Apply readout inversion to the folded counts
    emp_m = np.zeros(2**n)
    for bs, c in c_m.items():
        emp_m[int(bs, 2)] = c / shots
    q_m = np.linalg.solve(C, emp_m)
    zne_readout_mit.append(float(q_m[popcount >= K].sum()))

# Linear Richardson extrapolation: E(0) = (lam2*E(lam1) - lam1*E(lam2)) / (lam2 - lam1)
# For lam1=1, lam2=3: E(0) = (3*E(1) - E(3)) / 2
zne_only_est = (3.0 * zne_raw[0] - zne_raw[1]) / 2.0
layered_mit_est = (3.0 * zne_readout_mit[0] - zne_readout_mit[1]) / 2.0

print(f"ZNE raw estimates by scale {scale_factors}: {[round(x, 4) for x in zne_raw]}")
print(f"ZNE gate-mitigated estimate : {zne_only_est:.4f}")
print(f"Layered (Readout + ZNE)     : {layered_mit_est:.4f} | exact: {ref['exact_probability']:.4f}")

# ---- transpiled circuit statistics ----
def stats(tc, label):
    ops = tc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz", "swap"))
    one_q = sum(v for k, v in ops.items() if k not in ("cx", "cz", "ecr", "rzz", "swap", "measure", "barrier"))
    print(f"{label}: depth={tc.depth()} 1q={one_q} 2q={two_q} swaps={ops.get('swap', 0)} ops={dict(ops)}")


stats(qc, "original (logical)")
stats(transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=CouplingMap.from_line(n),
                optimization_level=1, seed_transpiler=42), "transpiled to rz/sx/x/cx on a line")
