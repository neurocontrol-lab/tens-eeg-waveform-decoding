"""
task3_erp_stats.py
==================
Statistics for comparing ERP peaks across conditions (waveforms or amplitudes).

- rm_anova           one-way repeated-measures ANOVA (within-subject factor)
- friedman_test      non-parametric counterpart
- posthoc_pairwise   paired t / Wilcoxon with Holm correction + Cohen's dz
- cluster_test_oneway  MNE cluster-based permutation F-test across conditions

The ANOVA is computed directly from sums of squares so the pure-stats functions
have no statsmodels dependency and stay trivially testable.  cluster_test_oneway
wraps mne.stats and is imported lazily so the rest of the module loads without MNE.
"""

import numpy as np
from scipy import stats as sps


def rm_anova(data):
    """One-way repeated-measures ANOVA.

    Parameters
    ----------
    data : (n_subjects, n_conditions) array, balanced, no missing cells.

    Returns
    -------
    dict with F, p, eta2_partial, df1 (condition), df2 (error).
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    grand = data.mean()
    subj_means = data.mean(axis=1, keepdims=True)   # (n,1)
    cond_means = data.mean(axis=0, keepdims=True)    # (1,k)

    ss_cond = n * ((cond_means - grand) ** 2).sum()
    ss_subj = k * ((subj_means - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_err = ss_total - ss_cond - ss_subj

    df_cond = k - 1
    df_err = (k - 1) * (n - 1)
    ms_cond = ss_cond / df_cond if df_cond > 0 else np.nan
    ms_err = ss_err / df_err if df_err > 0 else np.nan
    F = ms_cond / ms_err if (ms_err and ms_err > 0) else np.nan
    p = float(sps.f.sf(F, df_cond, df_err)) if np.isfinite(F) else np.nan
    eta2p = ss_cond / (ss_cond + ss_err) if (ss_cond + ss_err) > 0 else np.nan
    return {"F": float(F) if np.isfinite(F) else np.nan,
            "p": p,
            "eta2_partial": float(eta2p) if np.isfinite(eta2p) else np.nan,
            "df1": int(df_cond), "df2": int(df_err)}


def friedman_test(data):
    """Friedman test across conditions. data: (n_subjects, n_conditions).
    Returns (statistic, p_value)."""
    data = np.asarray(data, dtype=float)
    cols = [data[:, j] for j in range(data.shape[1])]
    stat, p = sps.friedmanchisquare(*cols)
    return float(stat), float(p)


def _holm(pvals):
    """Holm step-down adjusted p-values, preserving input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)   # enforce monotonic non-decreasing
        adj[idx] = running
    return adj


def posthoc_pairwise(data, labels, parametric=True):
    """All pairwise within-subject comparisons with Holm correction.

    Parameters
    ----------
    data : (n_subjects, n_conditions) array
    labels : list[str] of length n_conditions
    parametric : paired t-test if True, Wilcoxon signed-rank if False

    Returns
    -------
    list of dict: {a, b, stat, dz, p_raw, p_holm}, one per condition pair.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    pairs, raw = [], []
    for i in range(k):
        for j in range(i + 1, k):
            xi, xj = data[:, i], data[:, j]
            if parametric:
                stat, p = sps.ttest_rel(xi, xj)
            else:
                try:
                    stat, p = sps.wilcoxon(xi, xj)
                except ValueError:           # all-zero differences
                    stat, p = np.nan, 1.0
            d = xi - xj
            dz = d.mean() / (d.std(ddof=1) + 1e-12)
            pairs.append((labels[i], labels[j], float(stat) if np.isfinite(stat) else np.nan, float(dz)))
            raw.append(float(p))
    adj = _holm(raw)
    out = []
    for (a, b, stat, dz), pr, pa in zip(pairs, raw, adj):
        out.append({"a": a, "b": b, "stat": stat, "dz": dz,
                    "p_raw": pr, "p_holm": float(pa)})
    return out


def cluster_test_oneway(condition_arrays, n_permutations=2000, seed=42):
    """Omnibus cluster-based permutation F-test across >=2 conditions.

    Parameters
    ----------
    condition_arrays : list of (n_subjects, n_times) arrays, one per condition.

    Returns
    -------
    (F_obs, clusters, cluster_pv) from mne.stats.permutation_cluster_test.
    clusters is a list of boolean masks over the time axis; cluster_pv the
    corresponding p-values.
    """
    import mne  # lazy import keeps the module MNE-free for unit tests
    F_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_test(
        condition_arrays, n_permutations=n_permutations, tail=1,
        seed=seed, out_type="mask", verbose=False,
    )
    return F_obs, clusters, cluster_pv


def cluster_test_rm(arr_a, arr_b, n_permutations=1000, seed=42):
    """Repeated-measures cluster test for two conditions.

    Correct small-n within-subject test: a 1-sample permutation cluster test on
    the per-subject paired differences (A - B).

    Parameters
    ----------
    arr_a, arr_b : (n_subjects, n_times) arrays, same subjects in same order.

    Returns
    -------
    (t_obs, clusters, cluster_pv) from mne.stats.permutation_cluster_1samp_test.
    """
    import mne
    diff = np.asarray(arr_a, dtype=float) - np.asarray(arr_b, dtype=float)
    t_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
        diff, n_permutations=n_permutations, seed=seed,
        out_type="mask", verbose=False,
    )
    return t_obs, clusters, cluster_pv
