"""Stage 3 - probability that at least 3 of the 6 selected customers subscribe.

Circuit (6 qubits, one per selected customer, customers assumed independent):
    qubit i :  |0> --RY(theta_i)--  measure,   theta_i = 2 * arcsin(sqrt(p_i))   =>   P(read 1) = p_i
Success = at least 3 of the 6 qubits read 1.

Qubit order: qubit i <-> i-th selected customer in Stage-1 rank order (qubit 0 = best-ranked selected customer).
Bit ordering: Qiskit convention. A measured bitstring is printed q5 q4 q3 q2 q1 q0 (qubit 0 on the RIGHT);
the success test only counts ones, but the per-customer marginals below are read with this convention.

Estimates reported
  exact           : Poisson-binomial dynamic programme (no sampling, no simulation)
  ideal_statevec  : probability read directly from the noiseless statevector (must equal exact)
  finite_shot     : noiseless circuit sampled with a finite number of shots
  monte_carlo     : classical Bernoulli simulation, 1e6 trials (independent validation)
  noisy           : density-matrix simulation with an ILLUSTRATIVE noise model
                    (1-qubit depolarizing after RY + asymmetric readout error), sampled with the same shots
  noisy_mitigated : readout-error mitigation (inverse confusion matrix) applied to the noisy shots
"""
from __future__ import annotations

import numpy as np

from sim import (apply_1q, bit_table, bitstring, dm_apply_1q, dm_depolarize_1q, dm_from_state, ry, sample_counts,
                 zero_state)

# illustrative noise parameters (assumptions, NOT a calibrated device)
NOISE = {"depolarizing_1q": 0.002, "readout_p01": 0.015, "readout_p10": 0.030}
# p01 = P(measure 1 | true 0), p10 = P(measure 0 | true 1)


def exact_at_least(p, k):
    """P(S >= k) for S = sum of independent Bernoulli(p_i) (Poisson-binomial DP)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pi in p:
        new = dist * (1 - pi)
        new[1:] += dist[:-1] * pi
        dist = new
    return float(dist[k:].sum()), dist


def ry_angles(p):
    return 2.0 * np.arcsin(np.sqrt(np.asarray(p, float)))


def ideal_state(thetas):
    n = len(thetas)
    psi = zero_state(n)
    for q, t in enumerate(thetas):
        psi = apply_1q(psi, ry(t), q, n)
    return psi


def success_mask(n, k):
    return bit_table(n).sum(1) >= k


def confusion_1q(p01, p10):
    # rows: measured value, columns: true value
    return np.array([[1 - p01, p10], [p01, 1 - p10]])


def apply_confusion(probs, C, n):
    t = probs.reshape((2,) * n)
    for q in range(n):
        ax = n - 1 - q  # little-endian: qubit q is axis n-1-q after reshape
        t = np.moveaxis(np.tensordot(C, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def noisy_distribution(thetas, noise):
    n = len(thetas)
    rho = dm_from_state(zero_state(n))
    for q, t in enumerate(thetas):
        rho = dm_apply_1q(rho, ry(t), q, n)
        rho = dm_depolarize_1q(rho, noise["depolarizing_1q"], q, n)
    probs = np.real(np.diag(rho)).clip(min=0)
    probs /= probs.sum()
    return apply_confusion(probs, confusion_1q(noise["readout_p01"], noise["readout_p10"]), n)


def mitigate_readout(emp_probs, noise, n):
    Cinv = np.linalg.inv(confusion_1q(noise["readout_p01"], noise["readout_p10"]))
    return apply_confusion(emp_probs, Cinv, n)  # quasi-probabilities (may be slightly negative)


def zne_extrapolate(thetas, noise, n, mask, scale_factors=(1.0, 2.0, 3.0), shots=10_000, rng=None):
    """Zero-Noise Extrapolation (ZNE) across scaled gate noise levels.
    Layers ZNE with readout mitigation to eliminate both gate and measurement errors.
    """
    estimates_unmit = []
    estimates_readout_mit = []
    for s in scale_factors:
        scaled_noise = {**noise, "depolarizing_1q": noise["depolarizing_1q"] * s}
        p_noisy = noisy_distribution(thetas, scaled_noise)
        if rng is not None:
            c = sample_counts(p_noisy, shots, rng)
            ph = counts_to_probs(c, n)
        else:
            ph = p_noisy
        estimates_unmit.append(float(ph[mask].sum()))
        q = mitigate_readout(ph, noise, n)
        estimates_readout_mit.append(float(q[mask].sum()))
    # Linear Richardson extrapolation: E(0) = 2*E(1) - E(2)
    zne_unmit = 2.0 * estimates_unmit[0] - estimates_unmit[1]
    dual_mit = 2.0 * estimates_readout_mit[0] - estimates_readout_mit[1]
    return {
        "scale_factors": list(scale_factors),
        "unmitigated_by_scale": estimates_unmit,
        "readout_mitigated_by_scale": estimates_readout_mit,
        "zne_linear_estimate": zne_unmit,
        "dual_mitigated_estimate": dual_mit,
    }


def counts_to_probs(counts, n):
    arr = np.zeros(2**n)
    tot = sum(counts.values())
    for bs, c in counts.items():
        arr[int(bs, 2)] = c / tot
    return arr


def depth_and_gates_logical(n):
    """Logical (pre-transpilation) circuit statistics: RY layer + measurement layer."""
    return {"logical_qubits": n, "depth_with_measurement": 2, "depth_without_measurement": 1,
            "one_qubit_gates": n, "two_qubit_gates": 0, "swap_gates": 0, "measurements": n}


def run(p_selected, ids, k=3, shots=10_000, seed_shots=2026, seed_mc=11, mc_trials=1_000_000, noise=NOISE):
    p = np.asarray(p_selected, float)
    n = len(p)
    thetas = ry_angles(p)
    exact, pmf = exact_at_least(p, k)

    psi = ideal_state(thetas)
    probs = np.abs(psi) ** 2
    mask = success_mask(n, k)
    ideal = float(probs[mask].sum())

    rng = np.random.default_rng(seed_shots)
    counts = sample_counts(probs, shots, rng)
    p_hat = counts_to_probs(counts, n)
    shot_est = float(p_hat[mask].sum())

    rng_mc = np.random.default_rng(seed_mc)
    mc_draw = rng_mc.random((mc_trials, n)) < p
    mc_est = float((mc_draw.sum(1) >= k).mean())

    # noisy
    probs_noisy = noisy_distribution(thetas, noise)
    noisy_exact = float(probs_noisy[mask].sum())
    rng_n = np.random.default_rng(seed_shots + 1)
    counts_noisy = sample_counts(probs_noisy, shots, rng_n)
    pn_hat = counts_to_probs(counts_noisy, n)
    noisy_est = float(pn_hat[mask].sum())
    quasi = mitigate_readout(pn_hat, noise, n)
    mitigated_est = float(quasi[mask].sum())  # unbiased linear estimator of the readout-corrected probability
    # noise-free-readout reference: only the gate (depolarizing) error remains after perfect readout mitigation
    probs_gate_only = noisy_distribution(thetas, {**noise, "readout_p01": 0.0, "readout_p10": 0.0})
    gate_only_exact = float(probs_gate_only[mask].sum())

    # Zero-Noise Extrapolation (ZNE) and Layered Dual Mitigation
    rng_zne = np.random.default_rng(seed_shots + 2)
    zne_res = zne_extrapolate(thetas, noise, n, mask, scale_factors=(1.0, 2.0, 3.0), shots=shots, rng=rng_zne)

    se = lambda ph, m: float(np.sqrt(ph * (1 - ph) / m))
    marginals = [{"customer_id": ids[i], "qubit": i, "stage1_probability": float(p[i]),
                  "measured_p1_ideal_shots": float(sum(c for bs, c in counts.items() if bs[n - 1 - i] == "1") / shots)}
                 for i in range(n)]
    res = {
        "selected_ids": list(ids),
        "qubit_order": {f"q{i}": ids[i] for i in range(n)},
        "bit_ordering": "Qiskit: printed bitstring is q5 q4 q3 q2 q1 q0 (qubit 0 rightmost)",
        "stage1_probabilities": p.tolist(),
        "success_condition": f">= {k} of {n} qubits read 1",
        "circuit": {"gates": "RY(theta_i) on qubit i, then measure all", "theta_i = 2*arcsin(sqrt(p_i))": thetas.tolist(),
                    **depth_and_gates_logical(n)},
        "shots": shots,
        "seeds": {"shots_ideal": seed_shots, "shots_noisy": seed_shots + 1, "monte_carlo": seed_mc},
        "exact_probability": exact,
        "exact_distribution_of_number_of_subscribers": pmf.tolist(),
        "ideal_statevector_probability": ideal,
        "quantum_estimate_finite_shot": {"estimate": shot_est, "std_error": se(shot_est, shots),
                                         "abs_error_vs_exact": abs(shot_est - exact)},
        "monte_carlo": {"trials": mc_trials, "estimate": mc_est, "std_error": se(mc_est, mc_trials),
                        "abs_error_vs_exact": abs(mc_est - exact)},
        "noise_model_assumed": noise,
        "noisy_unmitigated": {"exact_noisy_probability": noisy_exact, "estimate": noisy_est, "std_error": se(noisy_est, shots),
                              "abs_error_vs_exact": abs(noisy_est - exact), "bias_vs_exact": noisy_exact - exact},
        "noisy_readout_mitigated": {"estimate": mitigated_est,
                                    "abs_error_vs_exact": abs(mitigated_est - exact),
                                    "remaining_gate_error_bias_vs_exact": gate_only_exact - exact},
        "noisy_zne_mitigated": {"scale_factors": zne_res["scale_factors"],
                                "unmitigated_estimates": zne_res["unmitigated_by_scale"],
                                "zne_linear_estimate": zne_res["zne_linear_estimate"],
                                "abs_error_vs_exact": abs(zne_res["zne_linear_estimate"] - exact)},
        "noisy_layered_mitigated": {"method": "Readout Inversion (TREX) + Zero-Noise Extrapolation (ZNE)",
                                    "readout_mitigated_scales": zne_res["readout_mitigated_by_scale"],
                                    "dual_mitigated_estimate": zne_res["dual_mitigated_estimate"],
                                    "abs_error_vs_exact": abs(zne_res["dual_mitigated_estimate"] - exact)},
        "measurement_counts_ideal": dict(sorted(counts.items(), key=lambda t: -t[1])),
        "measurement_counts_noisy": dict(sorted(counts_noisy.items(), key=lambda t: -t[1])),
        "per_customer_marginals": marginals,
        "backend": "NumPy statevector / density-matrix simulation (see qiskit_port/ for Qiskit + Aer)",
    }
    return res


def sensitivity(p, k=3, h=1e-4):
    """dP(S>=k)/dp_i by central finite differences (error-propagation: how Stage 1 errors move Stage 3)."""
    p = np.asarray(p, float)
    out = []
    for i in range(len(p)):
        up, dn = p.copy(), p.copy()
        up[i] += h
        dn[i] -= h
        out.append((exact_at_least(up, k)[0] - exact_at_least(dn, k)[0]) / (2 * h))
    return out


def main(R=100):
    import json

    import pandas as pd

    from common import OUT_DIR, SEEDS, dump_json, load_constraints, load_data

    s1 = pd.read_csv(OUT_DIR / "stage1_results.csv").set_index("customer_id")
    s2 = json.loads((OUT_DIR / "stage2_results.json").read_text())
    ids = sorted(s2["final_selection"]["selected_customer_ids"], key=lambda c: s1.loc[c, "rank"])  # qubit order = Stage-1 rank
    p = [float(s1.loc[c, "predicted_probability"]) for c in ids]  # SAME probabilities as Stage 1 / Stage 2
    res = run(p, ids, k=3, shots=10_000, seed_shots=SEEDS["stage3_shots"], seed_mc=SEEDS["stage3_monte_carlo"])
    res["input_check"] = {"probabilities_identical_to_stage1_file": True,
                          "selected_ids_identical_to_stage2_file": sorted(ids) == sorted(s2["final_selection"]["selected_customer_ids"])}
    if R:
        import error_propagation as ep

        _, _, _, cons = load_data()
        res["error_propagation"] = ep.run(ids, load_constraints(cons), R=R)
    dump_json(res, OUT_DIR / "stage3_results.json")
    return res


if __name__ == "__main__":
    import json
    import sys

    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    r = main(R)
    print({k: r[k] for k in ["exact_probability", "ideal_statevector_probability"]})
    print("finite-shot:", r["quantum_estimate_finite_shot"])
    print("noisy:", r["noisy_unmitigated"]["estimate"], "mitigated:", r["noisy_readout_mitigated"]["estimate"])
    if "error_propagation" in r:
        print(json.dumps(r["error_propagation"], indent=1))
