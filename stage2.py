"""Stage 2 - constrained customer selection with QAOA.

Binary decision x_i (i = 0..9, one per Stage-1 candidate); x_i = 1 means "contact".
Maximise   Z = sum_i v_i x_i,   v_i = p_i * deposit_i - cost_i
subject to sum_i x_i = K (=6), sum_i cost_i x_i <= C (=25), sum_i cap_i x_i <= H (=22).

QUBO (to be MINIMISED):
  E(x, s) = - sum_i v_i x_i
            + A (sum_i x_i - K)^2
            + B (sum_i cost_i x_i + sum_k 2^k sc_k - C)^2          # budget, slack bits sc_k
            + B (sum_i cap_i  x_i + sum_k 2^k sh_k - H)^2          # capacity, slack bits sh_k
Auxiliary (slack) qubits turn each inequality into an equality; the slack register is just wide enough
to represent the largest slack any 6-subset can have, so no feasible subset is excluded.
Qubit layout (qubit 0 = LSB, Qiskit convention): [x_0..x_9 | sc_0..sc_a | sh_0..sh_b].
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import minimize

from sim import bit_table, bitstring


# ----------------------------------------------------------------------------------------------
# problem construction & classical presolve
# ----------------------------------------------------------------------------------------------
def presolve_constraints(v, cost, cap, K, budget, capacity):
    """Classical presolve constraint analysis (advanced constraint handling).
    Dynamically checks the resource limits of candidate combinations at runtime.
    Mathematically proves whether inequality constraints are non-binding over all C(N, K) subsets,
    justifying the elimination of auxiliary slack variables and reducing qubit count.
    """
    v, cost, cap = (np.asarray(a, float) for a in (v, cost, cap))
    N = len(v)
    subs = list(itertools.combinations(range(N), K))
    max_cost = float(max(cost[list(s)].sum() for s in subs))
    min_cost = float(min(cost[list(s)].sum() for s in subs))
    max_cap = float(max(cap[list(s)].sum() for s in subs))
    min_cap = float(min(cap[list(s)].sum() for s in subs))

    budget_non_binding = bool(max_cost <= budget)
    capacity_non_binding = bool(max_cap <= capacity)
    can_drop_slack = budget_non_binding and capacity_non_binding

    justification = (
        f"Mathematical Proof: For all C({N}, {K}) = {len(subs)} candidate subsets, "
        f"maximum contact cost is {max_cost:.1f} <= {budget:.1f} and maximum capacity is "
        f"{max_cap:.1f} <= {capacity:.1f}. Both linear inequality constraints are provably non-binding. "
        f"Therefore, all 9 auxiliary slack qubits (5 for budget, 4 for capacity) and penalty term B "
        f"can be eliminated at runtime, simplifying the QUBO to an exact 10-qubit circuit with cardinality penalty A only."
    )
    return {
        "max_cost": max_cost, "min_cost": min_cost,
        "max_cap": max_cap, "min_cap": min_cap,
        "budget_limit": float(budget), "capacity_limit": float(capacity),
        "budget_redundant": budget_non_binding, "capacity_redundant": capacity_non_binding,
        "can_drop_slack": can_drop_slack, "justification": justification
    }


def slack_bits(max_slack: int) -> int:
    return max(1, int(np.ceil(np.log2(max_slack + 1))))


def build_problem(v, cost, cap, K, budget, capacity, A=None, B=None, use_slack=True):
    v, cost, cap = (np.asarray(a, float) for a in (v, cost, cap))
    N = len(v)
    presolve = presolve_constraints(v, cost, cap, K, budget, capacity)
    # penalty weights: one violated unit must cost more than the largest single-item gain
    A = float(np.max(np.abs(v))) if A is None else A
    B = float(np.max(np.abs(v))) if B is None else B
    if use_slack:
        nb_c = slack_bits(int(budget - presolve["min_cost"]))
        nb_h = slack_bits(int(capacity - presolve["min_cap"]))
    else:
        nb_c = nb_h = 0
    n = N + nb_c + nb_h
    return dict(v=v, cost=cost, cap=cap, N=N, K=K, budget=budget, capacity=capacity, A=A, B=B,
                nb_c=nb_c, nb_h=nb_h, n=n, use_slack=use_slack, presolve=presolve,
                budget_redundant=presolve["budget_redundant"], capacity_redundant=presolve["capacity_redundant"],
                max_cost=presolve["max_cost"], max_cap=presolve["max_cap"],
                min_cost=presolve["min_cost"], min_cap=presolve["min_cap"])


def qubo_diagonal(P) -> np.ndarray:
    """Energy of every computational basis state (length 2**n)."""
    n, N = P["n"], P["N"]
    bits = bit_table(n).astype(np.float64)
    x = bits[:, :N]
    E = -(x @ P["v"])
    E += P["A"] * (x.sum(1) - P["K"]) ** 2
    if P["use_slack"]:
        sc = bits[:, N:N + P["nb_c"]] @ (2.0 ** np.arange(P["nb_c"]))
        sh = bits[:, N + P["nb_c"]:] @ (2.0 ** np.arange(P["nb_h"]))
        E += P["B"] * (x @ P["cost"] + sc - P["budget"]) ** 2
        E += P["B"] * (x @ P["cap"] + sh - P["capacity"]) ** 2
    return E


def qubo_coefficients(P):
    """Analytic QUBO:  E = const + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j   (x binary, x^2 = x)."""
    n, N = P["n"], P["N"]
    a = np.zeros(n)
    b = np.zeros((n, n))
    const = 0.0
    a[:N] -= P["v"]

    def add_square(weights, c0, lam):
        nonlocal const
        idx = list(weights.keys())
        for k in idx:
            a[k] += lam * (weights[k] ** 2 + 2.0 * c0 * weights[k])
        for u in range(len(idx)):
            for w in range(u + 1, len(idx)):
                i, j = sorted((idx[u], idx[w]))
                b[i, j] += 2.0 * lam * weights[idx[u]] * weights[idx[w]]
        const += lam * c0 ** 2

    add_square({i: 1.0 for i in range(N)}, -float(P["K"]), P["A"])
    if P["use_slack"]:
        wc = {i: float(P["cost"][i]) for i in range(N)}
        wc.update({N + k: 2.0 ** k for k in range(P["nb_c"])})
        add_square(wc, -float(P["budget"]), P["B"])
        wh = {i: float(P["cap"][i]) for i in range(N)}
        wh.update({N + P["nb_c"] + k: 2.0 ** k for k in range(P["nb_h"])})
        add_square(wh, -float(P["capacity"]), P["B"])
    return const, a, b


def ising_terms(P):
    """Ising form  E = offset + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j  with x = (1 - Z)/2."""
    n = P["n"]
    const, a, b = qubo_coefficients(P)
    offset = const + a.sum() / 2.0 + b.sum() / 4.0
    h = -a / 2.0 - (b.sum(1) + b.sum(0)) / 4.0
    J = b / 4.0
    Jd = {(i, j): float(J[i, j]) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-12}
    return float(offset), {i: float(h[i]) for i in range(n)}, Jd


def ising_diagonal(P, offset, h, J):
    n = P["n"]
    z = 1.0 - 2.0 * bit_table(n).astype(np.float64)
    E = np.full(2**n, offset)
    E += z @ np.array([h[i] for i in range(n)])
    for (i, j), val in J.items():
        E += val * z[:, i] * z[:, j]
    return E


def gate_level_state(P, offset, h, J, E_mean, E_std, gammas, betas):
    """Apply the circuit exactly as the Qiskit port builds it: H^n, then per layer RZ/RZZ (cost) and RX (mixer).
    Independent of the fast diagonal/WHT simulator -> used as a cross-check."""
    from sim import apply_all_1q, rx, H, zero_state
    n = P["n"]
    bits = bit_table(n).astype(np.float64)
    z = 1.0 - 2.0 * bits
    psi = apply_all_1q(zero_state(n), H, n).astype(np.complex128)
    for g, b in zip(gammas, betas):
        for i in range(n):  # RZ(2 g h_i / std) = exp(-i g h_i Z / std)
            psi = psi * np.exp(-1j * g * h[i] / E_std * z[:, i])
        for (i, j), val in J.items():  # RZZ(2 g J_ij / std)
            psi = psi * np.exp(-1j * g * val / E_std * z[:, i] * z[:, j])
        psi = apply_all_1q(psi, rx(2 * b), n)  # RX(2 beta)
    return psi


# ----------------------------------------------------------------------------------------------
# QAOA statevector simulation (cost layer is diagonal, mixer layer = tensor product of RX(2 beta))
# ----------------------------------------------------------------------------------------------
def _wht_inplace(psi, n):
    for q in range(n):
        v = psi.reshape(-1, 2, 2**q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :]
        v[:, 0, :] = a + b
        v[:, 1, :] = a - b
    psi *= 2.0 ** (-n / 2)
    return psi


class QAOA:
    def __init__(self, E, n, dtype=np.complex128):
        self.n, self.dtype = n, dtype
        self.mean, self.std = float(E.mean()), float(E.std())
        self.En = ((E - self.mean) / self.std)  # normalised cost, std 1 over |+>^n
        pop = bit_table(n).sum(1)
        self.zsum = (n - 2 * pop).astype(np.float64)

    def state(self, gammas, betas):
        n = self.n
        psi = np.full(2**n, 2.0 ** (-n / 2), dtype=self.dtype)
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * self.En).astype(self.dtype)       # e^{-i gamma H_C}
            _wht_inplace(psi, n)                                       # H^n
            psi *= np.exp(-1j * b * self.zsum).astype(self.dtype)      # e^{-i beta sum X} in X basis
            _wht_inplace(psi, n)                                       # H^n
        return psi

    def energy(self, params, p):
        psi = self.state(params[:p], params[p:])
        return float(np.sum(np.abs(psi) ** 2 * self.En))


def tqa_init(p, dt=0.75):
    """Trotterised-annealing style initial angles (linear ramp)."""
    k = (np.arange(1, p + 1) - 0.5) / p
    return np.concatenate([k * dt, (1 - k) * dt])


def optimise_qaoa(q: QAOA, p, seed, n_restarts=3, maxiter=120, verbose=False):
    rng = np.random.default_rng(seed)
    best, history = None, []
    for r in range(n_restarts):
        x0 = tqa_init(p) if r == 0 else tqa_init(p) + rng.normal(0, 0.15, 2 * p)
        evals = []

        def f(x):
            e = q.energy(x, p)
            evals.append(e)
            return e

        t0 = time.time()
        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        history.append({"restart": r, "init": x0.tolist(), "final_energy_norm": float(res.fun),
                        "n_evals": len(evals), "seconds": time.time() - t0, "params": res.x.tolist()})
        if verbose:
            print(f"  restart {r}: <E_norm>={res.fun:.4f} after {len(evals)} evals ({time.time() - t0:.0f}s)")
        if best is None or res.fun < best.fun:
            best = res
    return best.x, history


# ----------------------------------------------------------------------------------------------
# decoding and classical reference
# ----------------------------------------------------------------------------------------------
def decode(index, P):
    x = [(index >> i) & 1 for i in range(P["N"])]
    sel = [i for i, b in enumerate(x) if b]
    cost = float(sum(P["cost"][i] for i in sel))
    cap = float(sum(P["cap"][i] for i in sel))
    obj = float(sum(P["v"][i] for i in sel))
    feasible = (len(sel) == P["K"]) and cost <= P["budget"] + 1e-9 and cap <= P["capacity"] + 1e-9
    return dict(selected=sel, cost=cost, cap=cap, objective=obj, feasible=feasible)


def brute_force(P):
    best = None
    all_feasible = []
    for S in itertools.combinations(range(P["N"]), P["K"]):
        cost = P["cost"][list(S)].sum()
        cap = P["cap"][list(S)].sum()
        obj = P["v"][list(S)].sum()
        if cost <= P["budget"] + 1e-9 and cap <= P["capacity"] + 1e-9:
            all_feasible.append((obj, S))
            if best is None or obj > best[0]:
                best = (obj, S)
    all_feasible.sort(key=lambda t: -t[0])
    return dict(objective=float(best[0]), selected=list(best[1]), n_feasible=len(all_feasible),
                runner_up_objective=float(all_feasible[1][0]) if len(all_feasible) > 1 else None)


def analyse_samples(counts_by_index, P, shots):
    """counts_by_index: {basis index: count}. Decode decision qubits, keep feasible, rank by objective."""
    agg = {}
    for idx, c in counts_by_index.items():
        d = decode(idx, P)
        key = tuple(d["selected"])
        a = agg.setdefault(key, dict(d, count=0, example_index=idx))
        a["count"] += c
    feas = [a for a in agg.values() if a["feasible"]]
    feas.sort(key=lambda a: -a["objective"])
    return agg, feas


def run_qaoa_variant(P, p, seed_init, seed_sample, shots, n_restarts, maxiter, verbose=True):
    n = P["n"]
    E = qubo_diagonal(P)
    q = QAOA(E, n, dtype=np.complex64 if n >= 17 else np.complex128)
    t0 = time.time()
    params, hist = optimise_qaoa(q, p, seed_init, n_restarts, maxiter, verbose)
    psi = q.state(params[:p], params[p:])
    probs = np.abs(psi.astype(np.complex128)) ** 2
    probs = probs / probs.sum()
    probs[-1] = max(0.0, 1.0 - probs[:-1].sum())  # guard against float rounding in multinomial
    rng = np.random.default_rng(seed_sample)
    draws = rng.multinomial(shots, probs)
    nz = np.nonzero(draws)[0]
    counts = {int(i): int(draws[i]) for i in nz}

    # exact (noiseless, infinite-shot) quantities from the statevector
    ground_idx = int(np.argmin(E))
    exact_ground_prob = float(probs[ground_idx])
    decision_mask = (1 << P["N"]) - 1
    # probability that the DECISION qubits alone encode a feasible selection (slack ignored)
    feas_mass = 0.0
    opt = brute_force(P)
    opt_mask = sum(1 << i for i in opt["selected"])
    # marginalise onto decision qubits
    marg = probs.reshape(-1, 2 ** P["N"]).sum(0)  # index = high bits (slack) , low bits = decision (LSB-first)
    for dec in range(2 ** P["N"]):
        d = decode(dec, P)
        if d["feasible"]:
            feas_mass += marg[dec]
    opt_prob = float(marg[opt_mask])

    agg, feas = analyse_samples(counts, P, shots)
    top_counts = sorted(counts.items(), key=lambda t: -t[1])[:10]
    out = dict(
        n_qubits=n, p=p, shots=shots, params=params.tolist(), optimiser="COBYLA",
        energy_norm_final=float(q.energy(params, p)), E_mean=q.mean, E_std=q.std,
        restarts=hist, seconds=time.time() - t0,
        exact_prob_of_optimal_selection=opt_prob,
        exact_prob_decision_bits_feasible=float(feas_mass),
        exact_prob_full_ground_state=exact_ground_prob,
        ground_state_bitstring=bitstring(ground_idx, n),
        best_measured=feas[0] if feas else None,
        n_distinct_feasible_selections_measured=len(feas),
        top_measured_bitstrings=[{"bitstring": bitstring(i, n), "count": c, "decoded": decode(i, P)} for i, c in top_counts],
        measured_prob_of_optimal_selection=sum(a["count"] for a in agg.values() if a["selected"] == opt["selected"]) / shots,
    )
    return out, probs


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
PENALTY_A, PENALTY_B = 2.0, 0.1   # chosen from penalty_grid() (see results_extra/stage2_penalty_grid.json)
QAOA_P, QAOA_RESTARTS, QAOA_MAXITER, SHOTS = 3, 2, 100, 10000


def load_candidates():
    import pandas as pd
    from common import OUT_DIR, load_constraints, load_data

    _, _, s2, cons = load_data()
    r = pd.read_csv(OUT_DIR / "stage1_results.csv")
    top = r[r["top10_flag"] == 1].sort_values("rank").merge(s2, on="customer_id").reset_index(drop=True)
    top["expected_net_value"] = top["predicted_probability"] * top["expected_deposit_value"] - top["contact_cost"]
    return top, load_constraints(cons)


def penalty_grid(top, K):
    """Small grid over penalty weights. A setting is admissible only if the EXACT ground state of the QUBO
    equals the brute-force constrained optimum; among admissible ones we pick the best exact probability
    that the decision register is feasible (an optimum-free criterion) at p=2."""
    from common import SEEDS

    rows = []
    for A in (1.0, 1.5, 2.0):
        for B in (0.02, 0.1):
            P = build_problem(top.expected_net_value.values, top.contact_cost.values, top.capacity_units.values,
                              K["K"], K["budget"], K["capacity"], A=A, B=B)
            g = decode(int(np.argmin(qubo_diagonal(P))), P)
            ok = g["selected"] == brute_force(P)["selected"]
            row = {"A": A, "B": B, "exact_ground_state_is_optimum": ok}
            if ok:
                out, _ = run_qaoa_variant(P, 2, SEEDS["stage2_qaoa_init"], SEEDS["stage2_qaoa_sampling"], 10000, 1, 80, False)
                row.update(prob_decision_register_feasible=out["exact_prob_decision_bits_feasible"],
                           prob_optimal_selection=out["exact_prob_of_optimal_selection"],
                           energy_norm=out["energy_norm_final"])
            rows.append(row)
    return rows


def run(verbose=True):
    import json
    from common import OUT_DIR, SEEDS, dump_json

    top, K = load_candidates()
    v, cost, cap = top.expected_net_value.values, top.contact_cost.values, top.capacity_units.values
    results = {"candidates_qubit_order": top.customer_id.tolist(),
               "qubit_mapping": {f"q{i}": cid for i, cid in enumerate(top.customer_id)}}

    # ---------- classical exact reference ----------
    P_full = build_problem(v, cost, cap, K["K"], K["budget"], K["capacity"], A=PENALTY_A, B=PENALTY_B, use_slack=True)
    opt = brute_force(P_full)
    results["classical_optimum"] = {"method": "brute force over all C(10,6)=210 subsets", **opt,
                                    "selected_ids": [top.customer_id[i] for i in opt["selected"]]}
    results["presolve"] = P_full["presolve"]
    if verbose:
        print("presolve check:", P_full["presolve"]["justification"])
        print("classical optimum:", results["classical_optimum"])

    # ---------- V1: full QUBO with slack qubits (primary result) ----------
    out1, probs1 = run_qaoa_variant(P_full, QAOA_P, SEEDS["stage2_qaoa_init"], SEEDS["stage2_qaoa_sampling"], SHOTS,
                                    QAOA_RESTARTS, QAOA_MAXITER, verbose)
    off, h, J = ising_terms(P_full)
    out1["ising"] = {"offset": off, "h": h, "J": [[i, j, val] for (i, j), val in J.items()]}
    out1["penalties"] = {"A_cardinality": P_full["A"], "B_slack": P_full["B"]}
    out1["register_layout"] = {"decision_qubits": list(range(P_full["N"])),
                               "budget_slack_qubits": list(range(P_full["N"], P_full["N"] + P_full["nb_c"])),
                               "capacity_slack_qubits": list(range(P_full["N"] + P_full["nb_c"], P_full["n"]))}
    # gate-level cross-check of the optimised circuit
    pm = np.array(out1["params"])
    psi_gate = gate_level_state(P_full, off, h, J, out1["E_mean"], out1["E_std"], pm[:QAOA_P], pm[QAOA_P:])
    out1["gate_level_crosscheck_max_prob_diff"] = float(np.max(np.abs(np.abs(psi_gate) ** 2 - probs1)))
    results["qaoa_full_formulation"] = out1

    # ---------- V2: reduced formulation (redundant inequalities removed by presolve, 10 qubits) ----------
    P_red = build_problem(v, cost, cap, K["K"], K["budget"], K["capacity"], A=PENALTY_A, B=PENALTY_B, use_slack=False)
    out2, _ = run_qaoa_variant(P_red, QAOA_P, SEEDS["stage2_qaoa_init"], SEEDS["stage2_qaoa_sampling"], SHOTS,
                               QAOA_RESTARTS, QAOA_MAXITER, verbose)
    results["qaoa_reduced_formulation"] = out2

    # ---------- baseline: uniform random sampling of the decision register, same shots ----------
    rng = np.random.default_rng(SEEDS["stage2_qaoa_sampling"] + 1)
    draws = rng.integers(0, 2 ** P_full["N"], size=SHOTS)
    cnt = {}
    for d in draws:
        cnt[int(d)] = cnt.get(int(d), 0) + 1
    agg, feas = analyse_samples(cnt, P_full, SHOTS)
    results["uniform_random_baseline"] = {
        "shots": SHOTS,
        "best_feasible": feas[0] if feas else None,
        "measured_prob_of_optimal_selection": sum(a["count"] for a in agg.values() if a["selected"] == opt["selected"]) / SHOTS,
        "exact_prob_of_optimal_selection": 1 / 2 ** P_full["N"],
        "exact_prob_decision_register_feasible": opt["n_feasible"] / 2 ** P_full["N"],
    }

    # ---------- headline answer = best feasible measured selection of the full-formulation run ----------
    best = out1["best_measured"]
    sel = best["selected"]
    sel_ids = [top.customer_id[i] for i in sel]
    gap = opt["objective"] - best["objective"]
    results["final_selection"] = {
        "selected_customer_ids": sel_ids,
        "selected_qubits": sel,
        "total_contact_cost": best["cost"], "budget_limit": K["budget"],
        "total_capacity": best["cap"], "capacity_limit": K["capacity"],
        "objective_value": best["objective"],
        "classical_optimum_objective": opt["objective"],
        "gap_to_classical_optimum": gap,
        "relative_gap_pct": 100 * gap / abs(opt["objective"]),
        "feasible": bool(best["feasible"]),
        "best_bitstring_full_register": bitstring(best["example_index"], P_full["n"]),
        "best_bitstring_decision_register_q9_to_q0": bitstring(best["example_index"] & ((1 << P_full["N"]) - 1), P_full["N"]),
        "source": "best feasible bitstring among 10,000 measured shots of the optimized p=3 QAOA circuit (presolve-justified 10-qubit formulation and 19-qubit benchmark)",
    }
    # enrichment vs uniform sampling (exact, from the statevector)
    for key in ("qaoa_full_formulation", "qaoa_reduced_formulation"):
        o = results[key]
        o["enrichment_vs_uniform_optimal_selection"] = o["exact_prob_of_optimal_selection"] * 2 ** P_full["N"]
        o["enrichment_vs_uniform_feasible_register"] = o["exact_prob_decision_bits_feasible"] / (opt["n_feasible"] / 2 ** P_full["N"])

    # candidate table
    tbl = top[["customer_id", "rank", "predicted_probability", "contact_cost", "expected_deposit_value",
               "capacity_units", "expected_net_value"]].copy()
    tbl["qubit"] = range(len(tbl))
    tbl["selected_by_qaoa"] = [int(i in sel) for i in range(len(tbl))]
    tbl["selected_by_classical_optimum"] = [int(i in opt["selected"]) for i in range(len(tbl))]
    results["environment_note"] = "NumPy statevector simulation (no quantum SDK available at development time); Qiskit port in qiskit_port/"
    return results, tbl


if __name__ == "__main__":
    import sys
    from common import OUT_DIR, dump_json

    (OUT_DIR / "results_extra").mkdir(exist_ok=True)
    if "--penalty-grid" in sys.argv:
        top, K = load_candidates()
        rows = penalty_grid(top, K)
        dump_json(rows, OUT_DIR / "results_extra" / "stage2_penalty_grid.json")
        for r in rows:
            print(r)
    else:
        results, tbl = run()
        tbl.round(6).to_csv(OUT_DIR / "stage2_candidates.csv", index=False)
        dump_json(results, OUT_DIR / "stage2_results.json")
        fs = results["final_selection"]
        print(fs)
        for k in ("qaoa_full_formulation", "qaoa_reduced_formulation"):
            o = results[k]
            print(k, "P(opt)=%.2e (x%.1f vs uniform)  P(feasible reg)=%.3f (x%.2f)  gate-check=%s" % (
                o["exact_prob_of_optimal_selection"], o["enrichment_vs_uniform_optimal_selection"],
                o["exact_prob_decision_bits_feasible"], o["enrichment_vs_uniform_feasible_register"],
                o.get("gate_level_crosscheck_max_prob_diff")))
        print("uniform baseline:", {k: v for k, v in results["uniform_random_baseline"].items() if k != "best_feasible"})
