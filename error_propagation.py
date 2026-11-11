"""How do Stage 1 errors propagate into Stage 2 and Stage 3?

We refit the final Stage 1 pipeline (same model, same calibration) on R stratified 80% subsamples of the
TRAINING data (no labels of the test set exist or are used), re-predict the 100 test customers, and track
  * overlap of the new top-10 with the submitted top-10,
  * whether the Stage 2 classical optimum (on the new top-10) equals the submitted six customers,
  * the Stage 3 probability P(>=3 of 6) for the SUBMITTED six using the re-fitted probabilities,
and a delta-method estimate  Var(P) ~ sum_i (dP/dp_i)^2 Var(p_i)  for comparison.
80% subsampling (without replacement) is used instead of a bootstrap because duplicated rows would leak
across the inner calibration folds. It slightly overstates variance (smaller training sets).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from common import OUT_DIR, SEEDS, load_data
from stage1 import PlattCalibrated, candidate_models
from stage3 import exact_at_least, sensitivity


def classical_optimum_ids(top, s2, K):
    d = top.merge(s2, on="customer_id")
    d["v"] = d["p"] * d["expected_deposit_value"] - d["contact_cost"]
    best, best_set = -1e18, None
    for S in itertools.combinations(range(len(d)), K["K"]):
        cost, cap = d.contact_cost.values[list(S)].sum(), d.capacity_units.values[list(S)].sum()
        if cost <= K["budget"] and cap <= K["capacity"]:
            v = d.v.values[list(S)].sum()
            if v > best:
                best, best_set = v, S
    return set(d.customer_id.values[list(best_set)])


def run(selected_ids, K, R=100, seed=SEEDS["data_split_cv"], verbose=True):
    tr, te, s2, cons = load_data()
    X, y = tr.drop(columns=["y"]), tr["y"].values
    Xt = te.drop(columns=["customer_id"])
    meta = json.loads((OUT_DIR / "results_extra" / "stage1_meta.json").read_text())
    model = PlattCalibrated(candidate_models(meta["lr_C"], meta["qk_config"])[meta["chosen_model"]], 5, seed)
    base = pd.read_csv(OUT_DIR / "stage1_results.csv")
    base_top = set(base.loc[base.top10_flag == 1, "customer_id"])
    sel = list(selected_ids)
    idx = {c: i for i, c in enumerate(te.customer_id)}
    sss = StratifiedShuffleSplit(n_splits=R, train_size=0.8, random_state=seed)
    P_sel, overlap, same_six, six_overlap, p3_sub = [], [], [], [], []
    for r, (a, _) in enumerate(sss.split(X, y)):
        m = clone(model).fit(X.iloc[a], y[a])
        p = m.predict_proba(Xt)[:, 1]
        dfp = pd.DataFrame({"customer_id": te.customer_id.values, "p": p}).sort_values(["p", "customer_id"], ascending=[False, True])
        top = dfp.head(10)
        overlap.append(len(base_top & set(top.customer_id)))
        opt_ids = classical_optimum_ids(top, s2, K)
        same_six.append(set(sel) == opt_ids)
        six_overlap.append(len(set(sel) & opt_ids))
        ps = np.array([p[idx[c]] for c in sel])
        P_sel.append(ps)
        p3_sub.append(exact_at_least(ps, 3)[0])
        if verbose and (r + 1) % 25 == 0:
            print(f"  subsample {r + 1}/{R}", flush=True)
    P_sel = np.array(P_sel)
    p_base = base.set_index("customer_id").loc[sel, "predicted_probability"].values
    sens = np.array(sensitivity(p_base, 3))
    sd_p = P_sel.std(0, ddof=1)
    delta_sd = float(np.sqrt(np.sum((sens * sd_p) ** 2)))  # independence assumed (optimistic)
    p3 = np.array(p3_sub)
    return {
        "method": f"{R} stratified 80% training subsamples; refit final pipeline incl. calibration; predict test again",
        "baseline_selected_ids": sel,
        "baseline_probabilities": p_base.tolist(),
        "top10_overlap_with_submitted": {"mean": float(np.mean(overlap)), "min": int(np.min(overlap)), "max": int(np.max(overlap))},
        "fraction_runs_same_six_selected": float(np.mean(same_six)),
        "mean_overlap_of_selected_six": float(np.mean(six_overlap)),
        "selected_probabilities_sd_across_subsamples": dict(zip(sel, sd_p.tolist())),
        "stage3_probability_baseline": exact_at_least(p_base, 3)[0],
        "stage3_probability_across_subsamples": {"mean": float(p3.mean()), "sd": float(p3.std(ddof=1)),
                                                 "p05": float(np.percentile(p3, 5)), "p95": float(np.percentile(p3, 95))},
        "delta_method_sd_stage3_assuming_independent_errors": delta_sd,
        "dP_dp_i": dict(zip(sel, sens.tolist())),
    }
