"""Qiskit + Aer port of the Stage 2 QAOA circuit (needs: pip install qiskit qiskit-aer pandas).

NOT executed in the environment where the results were produced (no quantum SDK installable there).
The gate list (H, RZ, RZZ, RX) is exactly the one used in ../src/stage2.py::gate_level_state, which was
verified in NumPy against the fast simulator to ~1e-11 in probability. Run this on your own machine; the
measured distribution should match ../stage2_results.json["qaoa_full_formulation"] up to shot noise.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "stage2_results.json").read_text())
q = res["qaoa_full_formulation"]
cand = pd.read_csv(ROOT / "stage2_candidates.csv")
presolve = res.get("presolve", {})
if presolve.get("can_drop_slack", False):
    print("Classical presolve check:", presolve.get("justification"))
n, p, shots = q["n_qubits"], q["p"], q["shots"]
params = np.array(q["params"])
gammas, betas = params[:p], params[p:]
sigma = q["E_std"]                                     # cost Hamiltonian is divided by this (normalisation)
h = {int(k): v for k, v in q["ising"]["h"].items()}
J = [(int(i), int(j), v) for i, j, v in q["ising"]["J"]]

qc = QuantumCircuit(n)
qc.h(range(n))
for g, b in zip(gammas, betas):
    for i, hi in h.items():
        qc.rz(2 * g * hi / sigma, i)                   # exp(-i g h_i Z_i / sigma)
    for i, j, v in J:
        qc.rzz(2 * g * v / sigma, i, j)                # exp(-i g J_ij Z_i Z_j / sigma)
    for i in range(n):
        qc.rx(2 * b, i)                                # exp(-i beta X_i)
qc.measure_all()

sim = AerSimulator(method="statevector", seed_simulator=7)
counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

N = len(cand)                                          # decision qubits 0..N-1 <-> candidate rows (Stage-1 rank order)
K, budget, capacity = 6, 25, 22
best = None
for bs, c in counts.items():
    x = [int(bs[n - 1 - i]) for i in range(N)]         # qubit i is character n-1-i of the printed string
    sel = [i for i in range(N) if x[i]]
    if len(sel) != K:
        continue
    cost = cand.contact_cost.values[sel].sum()
    cap = cand.capacity_units.values[sel].sum()
    if cost > budget or cap > capacity:
        continue
    obj = cand.expected_net_value.values[sel].sum()
    if best is None or obj > best[0]:
        best = (obj, sel)
print("best feasible measured selection:", [cand.customer_id[i] for i in best[1]], "objective", best[0])
print("classical optimum               :", res["classical_optimum"]["selected_ids"], res["classical_optimum"]["objective"])

# ---- transpiled statistics (QAOA is dense: 151 RZZ terms per layer, so expect many SWAPs on sparse hardware) ----
def stats(tc, label):
    ops = tc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz", "swap"))
    one_q = sum(v for k, v in ops.items() if k not in ("cx", "cz", "ecr", "rzz", "swap", "measure", "barrier"))
    print(f"{label}: depth={tc.depth()} 1q={one_q} 2q={two_q} swaps={ops.get('swap', 0)}")


stats(qc, "original (logical)")
stats(transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42), "rz/sx/x/cx, all-to-all")
stats(transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=CouplingMap.from_line(n),
                optimization_level=1, seed_transpiler=42), "rz/sx/x/cx, linear chain")
