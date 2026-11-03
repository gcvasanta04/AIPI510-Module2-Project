#!/usr/bin/env python3
"""
Analysis for the quantization experiment.

Reports
  1. Accuracy per level with Wilson 95% CIs
  2. Cochran's Q across all levels (omnibus, paired binary outcomes)
     + effect size eta^2_Q = Q / (n (k - 1))  (Serlin, Carr & Marascuilo, 1982)
  3. Exact McNemar for each level vs the reference (default q8_0), Holm-corrected,
     with effect sizes:
       - paired accuracy difference with Wald 95% CI
       - discordant odds ratio (ref-only / quant-only) with log-scale 95% CI
       - Cohen's g = (ref-only) / (discordant pairs) - 0.5
  4. Optional exploratory equivalence check (TOST via the 90% CI) for a margin
  5. Sensitivity analysis excluding items with any invalid (unparseable) output
  6. Prospective power analysis for McNemar (Connor, 1987) -- pilot data only.
     On pilot results (stage == 'pilot'), sections 2-4 and the figure are skipped.
     delta = --min-effect; psi = upper 95% Wilson bound of the observed discordance rate
  7. Worked calculations: every number in sections 1-4 with the values plugged in
     (main run only; --brief skips it)
  8. Figure: accuracy by level + paired difference vs reference
  9. CSV summary for the writeup tables
 10. Markdown report (<stage>_report.md): all of the above, with tables, next to the CSVs

Usage
  # pilot: size the main run
  python analyze.py results/pilot_results.jsonl --min-effect 0.05

  # main run: tests, effect sizes, figure
  python analyze.py results/main_results.jsonl --min-effect 0.05
  python analyze.py results/main_results.jsonl --min-effect 0.05 --equivalence-margin 0.05
"""

import argparse
import contextlib
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import binom, chi2, norm
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

LEVEL_ORDER = ["q2_K", "q3_K", "q3_K_S", "q3_K_M", "q3_K_L", "q4_0", "q4_K_S", "q4_K_M",
               "q5_K_S", "q5_K_M", "q6_K", "q8_0", "fp16"]
PILOT_MAX_N = 100  # samples at or below this size are treated as a pilot


# ---------------------------------------------------------------- statistics

def mcnemar_n(p10: float, p01: float, alpha: float, power: float) -> float:
    """Pairs needed for a two-sided McNemar test (Connor, 1987)."""
    psi, delta = p10 + p01, abs(p10 - p01)
    if delta == 0:
        return math.inf
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta**2)) ** 2 / delta**2


def fmt_n(x: float) -> str:
    return "inf (no observed difference)" if math.isinf(x) else str(math.ceil(x))


def paired_diff_ci(b: int, c: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Difference in paired proportions (quant - ref) with a Wald CI.

    b = ref right / quant wrong, c = ref wrong / quant right.
    diff = (c - b) / n ; SE = sqrt((b + c) - (b - c)^2 / n) / n
    """
    diff = (c - b) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
    z = norm.ppf(1 - (1 - conf) / 2)
    return diff, diff - z * se, diff + z * se


def discordant_or(b: int, c: int, conf: float = 0.95) -> tuple[float, float, float, bool]:
    """Conditional (discordant) odds ratio b/c with a log-scale CI.

    Adds 0.5 to both cells if either is zero (Haldane correction); flags when it did.
    """
    corrected = b == 0 or c == 0
    if corrected:
        b, c = b + 0.5, c + 0.5
    or_ = b / c
    se = math.sqrt(1 / b + 1 / c)
    z = norm.ppf(1 - (1 - conf) / 2)
    return or_, math.exp(math.log(or_) - z * se), math.exp(math.log(or_) + z * se), corrected


def pairwise_vs_reference(wide: pd.DataFrame, reference: str, others: list[str]) -> pd.DataFrame:
    n = len(wide)
    ref = wide[reference]
    rows = []
    for lvl in others:
        cur = wide[lvl]
        a = int(((ref == 1) & (cur == 1)).sum())
        b = int(((ref == 1) & (cur == 0)).sum())  # ref right, quantized wrong
        c = int(((ref == 0) & (cur == 1)).sum())  # ref wrong, quantized right
        d = int(((ref == 0) & (cur == 0)).sum())
        p = mcnemar([[a, b], [c, d]], exact=True).pvalue
        diff, lo, hi = paired_diff_ci(b, c, n)
        _, lo90, hi90 = paired_diff_ci(b, c, n, conf=0.90)
        or_, or_lo, or_hi, corrected = discordant_or(b, c)
        rows.append({
            "level": lvl, "n": n, "both_right": a, "ref_only": b, "quant_only": c, "both_wrong": d,
            "cohens_g": (b / (b + c) - 0.5) if b + c else 0.0,
            "acc_diff": diff, "diff_ci_lo": lo, "diff_ci_hi": hi,
            "diff_ci90_lo": lo90, "diff_ci90_hi": hi90,
            "disc_or": or_, "or_ci_lo": or_lo, "or_ci_hi": or_hi, "or_corrected": corrected,
            "p_raw": p,
        })
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res["p_raw"], method="holm")[1]
    return res


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def print_pairwise(res: pd.DataFrame, reference: str) -> None:
    print(f"  {'level':8s} {'Q8 only':>7s} {'Qk only':>7s}  {'diff (pts)':>10s}  {'95% CI':>16s}"
          f"  {'disc. OR':>8s}  {'95% CI':>14s}  {'g':>5s}  {'p (exact)':>9s}  {'p (Holm)':>9s}")
    for _, r in res.iterrows():
        flag = "*" if r["or_corrected"] else " "
        print(f"  {r['level']:8s} {r['ref_only']:7d} {r['quant_only']:7d}  "
              f"{100 * r['acc_diff']:+10.1f}  [{100 * r['diff_ci_lo']:+6.1f}, {100 * r['diff_ci_hi']:+6.1f}]"
              f"  {r['disc_or']:7.2f}{flag}  [{r['or_ci_lo']:5.2f}, {r['or_ci_hi']:5.2f}]"
              f"  {r['cohens_g']:5.2f}  {fmt_p(r['p_raw']):>9s}  {fmt_p(r['p_holm']):>9s}")
    print(f"  diff = accuracy(Qk) - accuracy({reference}), Wald CI.  disc. OR = ({reference} only) / (Qk only):"
          f"\n  how many times more often {reference} is the one that's right when the two disagree."
          f"\n  g = Cohen's g = ({reference} only) / (all disagreements) - 0.5; 0 = even split.")
    if res["or_corrected"].any():
        print("  * a zero cell; 0.5 added to both discordant counts for the OR")


def fmt_small(p: float) -> str:
    return f"{p:.4g}"


def worked_calculations(wide: pd.DataFrame, levels: list[str], res: pd.DataFrame, reference: str,
                        equivalence_margin: float | None) -> None:
    """Print every statistic from sections 1-4 with its inputs plugged into the formula."""
    n = len(wide)
    z95, z90 = norm.ppf(0.975), norm.ppf(0.95)
    print("\n=== 7. Worked calculations ===")

    # -- 7a. accuracy + Wilson CI
    print(f"\n  7a. Accuracy with Wilson 95% CI   (n = {n}, z = {z95:.4f})")
    print("      p_hat  = correct / n")
    print("      center = (p_hat + z^2/(2n)) / (1 + z^2/n)")
    print("      half   = z * sqrt(p_hat(1 - p_hat)/n + z^2/(4n^2)) / (1 + z^2/n)")
    print("      CI     = center +/- half")
    for lvl in levels:
        k = int(wide[lvl].sum())
        p = k / n
        den = 1 + z95**2 / n
        center = (p + z95**2 / (2 * n)) / den
        half = z95 * math.sqrt(p * (1 - p) / n + z95**2 / (4 * n * n)) / den
        print(f"      {lvl:8s} p_hat = {k}/{n} = {p:.4f}   center = {center:.4f}   half = {half:.4f}"
              f"   -> [{center - half:.3f}, {center + half:.3f}]")

    # -- 7b. Cochran's Q
    k_lv = len(levels)
    if k_lv > 2:
        col = wide.sum(axis=0)          # C_j: items correct at each level
        row = wide.sum(axis=1)          # R_i: levels correct for each item (0..k)
        N = int(col.sum())
        sum_c2 = int((col**2).sum())
        sum_r2 = int((row**2).sum())
        num = (k_lv - 1) * (k_lv * sum_c2 - N**2)
        den = k_lv * N - sum_r2
        Q = num / den
        p = chi2.sf(Q, k_lv - 1)
        print(f"\n  7b. Cochran's Q   (k = {k_lv} levels, n = {n} items)")
        print("      C_j = items correct at level j;  R_i = levels correct on item i;  N = sum of all C_j")
        print("      Q = (k - 1) * (k * sum(C_j^2) - N^2) / (k * N - sum(R_i^2))")
        print("      C_j: " + ", ".join(f"{l} = {int(col[l])}" for l in levels))
        print(f"      N = {' + '.join(str(int(col[l])) for l in levels)} = {N}")
        print(f"      sum(C_j^2) = {sum_c2}      sum(R_i^2) = {sum_r2}")
        dist = row.value_counts().sort_index()
        print("      R_i distribution (levels correct -> items): "
              + ", ".join(f"{int(r)} -> {int(v)}" for r, v in dist.items()))
        print(f"      Q = ({k_lv} - 1) * ({k_lv} * {sum_c2} - {N}^2) / ({k_lv} * {N} - {sum_r2})")
        print(f"        = {k_lv - 1} * {k_lv * sum_c2 - N**2} / {den} = {Q:.2f}")
        print(f"      df = k - 1 = {k_lv - 1};  p = P(chi-square({k_lv - 1}) >= {Q:.2f}) = {fmt_small(p)}")
        print(f"      eta^2_Q = Q / (n(k - 1)) = {Q:.2f} / ({n} * {k_lv - 1}) = {Q / (n * (k_lv - 1)):.4f}")

    # -- 7c. each comparison
    for _, r in res.iterrows():
        a, b, c, d = int(r["both_right"]), int(r["ref_only"]), int(r["quant_only"]), int(r["both_wrong"])
        lvl = r["level"]
        m = b + c
        print(f"\n  7c. {lvl} vs {reference}")
        print(f"      2x2 table ({n} items):")
        print(f"                        {lvl:>8s} right  {lvl:>8s} wrong")
        print(f"        {reference:>8s} right   a = {a:5d}        b = {b:5d}")
        print(f"        {reference:>8s} wrong   c = {c:5d}        d = {d:5d}")
        print(f"      Only the discordant cells matter: b + c = {b} + {c} = {m} disagreements.")

        # exact McNemar
        x = min(b, c)
        cdf = binom.cdf(x, m, 0.5) if m else 1.0
        p_exact = min(1.0, 2 * cdf)
        print("      Exact McNemar: under H0 each disagreement is a fair coin flip, X ~ Binomial(b + c, 0.5)")
        print(f"        p = 2 * P(X <= min(b, c)) = 2 * P(X <= {x} | {m}, 0.5) = 2 * {fmt_small(cdf)}"
              f" = {fmt_small(p_exact)}")

        # paired difference
        diff = (c - b) / n
        se = math.sqrt(max(m - (b - c) ** 2 / n, 0.0)) / n
        print("      Accuracy difference (Qk - ref):")
        print(f"        diff = (c - b) / n = ({c} - {b}) / {n} = {diff:+.5f}  ({100 * diff:+.2f} pts)")
        print(f"        SE   = sqrt((b + c) - (b - c)^2 / n) / n = sqrt({m} - {(b - c) ** 2}/{n}) / {n}"
              f" = {se:.5f}")
        print(f"        95% CI = {diff:+.5f} +/- {z95:.4f} * {se:.5f} = [{diff - z95 * se:+.4f}, {diff + z95 * se:+.4f}]")
        if equivalence_margin:
            print(f"        90% CI = {diff:+.5f} +/- {z90:.4f} * {se:.5f} = [{diff - z90 * se:+.4f}, "
                  f"{diff + z90 * se:+.4f}]   (TOST: inside +/-{equivalence_margin}?)")

        # discordant OR
        bb, cc = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
        or_ = bb / cc
        lse = math.sqrt(1 / bb + 1 / cc)
        print("      Discordant odds ratio:" + ("  (0.5 added to both cells: a zero count)" if bb != b else ""))
        print(f"        OR = b / c = {bb:g} / {cc:g} = {or_:.3f};   ln(OR) = {math.log(or_):.4f}")
        print(f"        SE(ln OR) = sqrt(1/b + 1/c) = sqrt(1/{bb:g} + 1/{cc:g}) = {lse:.4f}")
        print(f"        95% CI = exp({math.log(or_):.4f} +/- {z95:.4f} * {lse:.4f}) = "
              f"[{math.exp(math.log(or_) - z95 * lse):.3f}, {math.exp(math.log(or_) + z95 * lse):.3f}]")

        # Cohen's g
        if m:
            print(f"      Cohen's g = b / (b + c) - 0.5 = {b}/{m} - 0.5 = {b / m - 0.5:.3f}"
                  "   (0.05 small, 0.15 medium, 0.25 large; Cohen 1988)")

    # -- 7d. Holm
    mtot = len(res)
    order = res.sort_values("p_raw").reset_index(drop=True)
    print(f"\n  7d. Holm correction   (m = {mtot} comparisons vs {reference})")
    print("      Sort raw p-values ascending; multiply the i-th smallest by (m - i + 1);")
    print("      cap at 1, and never let an adjusted p be smaller than the one before it.")
    running = 0.0
    for i, r in order.iterrows():
        factor = mtot - i
        step = min(1.0, factor * r["p_raw"])
        running = max(running, step)
        note = "" if running == step else f"  -> raised to {fmt_small(running)} to keep the order"
        print(f"      {i + 1}. {r['level']:8s} {factor} * {fmt_small(r['p_raw'])} = {fmt_small(step)}{note}"
              f"   -> Holm p = {fmt_small(running)}")


def power_section(res: pd.DataFrame, n: int, args, is_pilot: bool) -> None:
    """Prospective McNemar sample size from the minimum effect and an estimated discordance rate.

    Effect size delta comes from --min-effect (a judgment, not the data). The data supply only the
    discordance rate psi; because a small sample estimates it imprecisely, the upper 95% Wilson
    bound is used as a conservative planning value (larger psi -> larger n).
    """
    alpha_strict = args.alpha / args.n_comparisons
    za, zb = norm.ppf(1 - alpha_strict / 2), norm.ppf(args.power)
    print(f"\n=== 6. Power analysis (two-sided McNemar, Connor 1987) ===")
    print(f"  alpha = {args.alpha}/{args.n_comparisons} = {alpha_strict:.4f}  ->  z(1-alpha/2) = {za:.4f}")
    print(f"  power = {args.power}                ->  z(1-beta)    = {zb:.4f}")
    d = args.min_effect
    if d:
        print(f"  minimum effect of interest delta = {d}")
        print("  n = [z(1-alpha/2)*sqrt(psi) + z(1-beta)*sqrt(psi - delta^2)]^2 / delta^2")

    for _, r in res.iterrows():
        k = int(r["ref_only"] + r["quant_only"])
        psi_hat = k / n
        psi_lo, psi_hi = proportion_confint(k, n, method="wilson")
        psi_plan = psi_hi
        print(f"\n  {r['level']} vs {args.reference}:")
        print(f"    discordant pairs: {int(r['ref_only'])} ({args.reference} only) + {int(r['quant_only'])} "
              f"({r['level']} only) = {k} of {n}")
        print(f"    psi_hat = {k}/{n} = {psi_hat:.3f}, 95% Wilson CI [{psi_lo:.3f}, {psi_hi:.3f}]")
        if not d:
            print("    Pass --min-effect (e.g. 0.05) to compute the required n.")
            continue
        if psi_plan < d:
            print(f"    ! upper bound {psi_plan:.3f} is below delta = {d}; a {d} effect needs psi >= delta, "
                  f"so planning with psi = {d}")
            psi_plan = d
        print(f"    planning psi = {psi_plan:.3f} (upper bound: conservative)")
        t1, t2 = za * math.sqrt(psi_plan), zb * math.sqrt(psi_plan - d**2)
        n_req = (t1 + t2) ** 2 / d**2
        print(f"    n = [{za:.4f}*sqrt({psi_plan:.4f}) + {zb:.4f}*sqrt({psi_plan:.4f} - {d**2:.4f})]^2 / {d**2:.4f}")
        print(f"      = [{t1:.4f} + {t2:.4f}]^2 / {d**2:.4f} = {(t1 + t2) ** 2:.4f} / {d**2:.4f} "
              f"= {n_req:.1f}  ->  n = {math.ceil(n_req)}")
        if psi_hat >= d:
            print(f"    (with the point estimate psi = {psi_hat:.3f} instead: "
                  f"n = {fmt_n(mcnemar_n((psi_hat + d) / 2, (psi_hat - d) / 2, alpha_strict, args.power))})")

    if is_pilot:
        print("\n  The pilot is used only for psi. delta is the minimum effect you chose, not the pilot's "
              "observed\n  difference, which is too noisy to plan with.")
    else:
        print("\n  Run on a full sample, this checks the planning assumption about psi; it is not post hoc power.")


# ---------------------------------------------------------------- markdown report

class _Tee(io.StringIO):
    """Writes to the real stdout and keeps a copy."""

    def __init__(self, stream):
        super().__init__()
        self._stream = stream

    def write(self, text):
        self._stream.write(text)
        return super().write(text)


def tee(fn, *args, **kwargs) -> str:
    """Run fn, printing as usual, and return what it printed."""
    buf = _Tee(sys.stdout)
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def md_code(text: str) -> str:
    lines = [l for l in text.splitlines() if not l.startswith("===")]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "```text\n" + "\n".join(lines).rstrip() + "\n```\n"


def md_pairwise(res: pd.DataFrame, reference: str) -> str:
    out = [f"| Comparison | Both right | {reference} only | Qk only | Both wrong | Diff (pts) | 95% CI (pts) "
           "| Disc. OR [95% CI] | Cohen's g | p (exact) | p (Holm) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        star = "\\*" if r["or_corrected"] else ""
        out.append(f"| {r['level']} vs {reference} | {r['both_right']} | {r['ref_only']} | {r['quant_only']} "
                   f"| {r['both_wrong']} | {100 * r['acc_diff']:+.1f} "
                   f"| [{100 * r['diff_ci_lo']:+.1f}, {100 * r['diff_ci_hi']:+.1f}] "
                   f"| {r['disc_or']:.2f}{star} [{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] "
                   f"| {r['cohens_g']:.2f} | {fmt_p(r['p_raw'])} | {fmt_p(r['p_holm'])} |")
    out.append("")
    out.append(f"- **Diff** = accuracy(Qk) − accuracy({reference}), with a Wald 95% CI (not adjusted for "
               "multiple comparisons).")
    out.append(f"- **Disc. OR** = ({reference} only) / (Qk only): how many times more often {reference} is the "
               "one that's right when the two models disagree. 1 = no difference.")
    out.append(f"- **Cohen's g** = ({reference} only) / (all disagreements) − 0.5. 0 = even split; "
               "0.05 small, 0.15 medium, 0.25 large (Cohen, 1988).")
    out.append("- **p (exact)** = exact McNemar; **p (Holm)** = Holm-adjusted across the comparisons.")
    if res["or_corrected"].any():
        out.append("- \\* a zero cell; 0.5 added to both discordant counts for the OR.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- figure

def make_figure(acc: pd.DataFrame, res: pd.DataFrame, reference: str, min_effect: float | None,
                path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface, blue = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    # left: accuracy per level with Wilson CIs
    x = range(len(acc))
    ax1.errorbar(x, 100 * acc["acc"], yerr=[100 * (acc["acc"] - acc["ci_lo"]), 100 * (acc["ci_hi"] - acc["acc"])],
                 fmt="o", color=blue, ecolor=blue, elinewidth=2, capsize=0, markersize=8,
                 markeredgecolor=surface, markeredgewidth=2)
    for i, a in enumerate(acc["acc"]):
        ax1.annotate(f"{100 * a:.1f}%", (i, 100 * a), xytext=(10, 0), textcoords="offset points",
                     va="center", fontsize=9, color=ink2)
    ax1.set_xticks(list(x), acc["level"])
    ax1.set_xlim(-0.5, len(acc) - 0.3)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Accuracy by quantization level (95% Wilson CI, n = {int(acc['n'].iloc[0])})",
                  fontsize=10, loc="left", color=ink)

    # right: paired difference vs reference
    y = range(len(res))
    ax2.set_ylim(-0.6, len(res) - 0.1)
    if min_effect:
        ax2.axvspan(-100 * min_effect, 100 * min_effect, color=grid, alpha=0.6, linewidth=0)
        ax2.text(0, len(res) - 0.3, f"±{100 * min_effect:g} pt minimum effect", fontsize=8, color=ink2,
                 ha="center", va="center", bbox={"facecolor": surface, "edgecolor": "none", "pad": 2})
    ax2.axvline(0, color=ink2, linewidth=1)
    ax2.errorbar(100 * res["acc_diff"], y,
                 xerr=[100 * (res["acc_diff"] - res["diff_ci_lo"]), 100 * (res["diff_ci_hi"] - res["acc_diff"])],
                 fmt="o", color=blue, ecolor=blue, elinewidth=2, capsize=0, markersize=8,
                 markeredgecolor=surface, markeredgewidth=2)
    # labels sit in their own column, clear of the CIs, the zero line and the band
    label_x = max(100 * res["diff_ci_hi"].max(), 100 * (min_effect or 0), 0) + 1.5
    for i, r in enumerate(res.itertuples()):
        ax2.text(label_x, i, f"{100 * r.acc_diff:+.1f} pts\nHolm p {fmt_p(r.p_holm)}",
                 va="center", fontsize=9, color=ink2)
    ax2.set_yticks(list(y), [f"{l} − {reference}" for l in res["level"]])
    lo = min(100 * res["diff_ci_lo"].min(), -100 * (min_effect or 0)) - 2
    ax2.set_xlim(lo, label_x + 9)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color=grid, linewidth=0.8)
    ax2.set_xlabel("Accuracy difference (percentage points)")
    ax2.set_title(f"Paired difference vs {reference} (95% CI)", fontsize=10, loc="left", color=ink)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--reference", default="q8_0")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--n-comparisons", type=int, default=3,
                    help="pairwise tests planned in the MAIN run (sets the Holm-strictest alpha)")
    ap.add_argument("--min-effect", type=float,
                    help="smallest accuracy drop you care about (e.g. 0.05); used for the power "
                         "analysis and drawn as a band on the figure")
    ap.add_argument("--equivalence-margin", type=float,
                    help="exploratory TOST: is the 90%% CI of each difference inside +/- margin?")
    ap.add_argument("--power-analysis", action="store_true",
                    help="force the power-analysis section on a large (non-pilot) sample")
    ap.add_argument("--brief", action="store_true", help="skip the worked-calculations section")
    ap.add_argument("--out-dir", help="where to write the figure and CSV (default: next to results)")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = results_path.stem.replace("_results", "")

    with results_path.open() as f:
        df = pd.DataFrame([json.loads(l) for l in f])
    if "valid" not in df:
        df["valid"] = True
    df["valid"] = df["valid"].astype(bool)

    levels = sorted(df["level"].unique(), key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 99)
    wide = df.pivot(index="item_id", columns="level", values="correct")[levels]
    n_before = len(wide)
    wide = wide.dropna().astype(int)
    if len(wide) < n_before:
        print(f"note: dropped {n_before - len(wide)} items missing a response at some level\n")
    n = len(wide)
    if "stage" in df:
        is_pilot = bool((df["stage"] == "pilot").all())
    else:  # older result files without a stage field
        is_pilot = n <= PILOT_MAX_N
    if is_pilot:
        print("Pilot run: hypothesis tests (sections 2-4), the figure and the pairwise CSV are skipped.\n"
              "The pilot is for checking the pipeline and estimating the discordance rate only.\n")

    md = [f"# Quantization experiment: {'pilot' if is_pilot else 'main'} analysis\n",
          f"- Results file: `{results_path}`",
          f"- Items analyzed: {n} (complete cases across {len(levels)} levels)",
          f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
          f"- Command: `python {' '.join(sys.argv)}`\n"]
    if is_pilot:
        md.append("> Pilot run: hypothesis tests, the figure and the pairwise CSV are skipped. The pilot is for "
                  "checking the pipeline and estimating the discordance rate only.\n")

    # ---- 1. accuracy
    print(f"=== 1. Accuracy (n = {n} items) ===")
    invalid = df[df["item_id"].isin(wide.index)].groupby("level")["valid"].apply(lambda s: int((~s).sum()))
    acc_rows = []
    for lvl in levels:
        k = int(wide[lvl].sum())
        lo, hi = proportion_confint(k, n, method="wilson")
        acc_rows.append({"level": lvl, "n": n, "correct": k, "acc": k / n, "ci_lo": lo, "ci_hi": hi,
                         "invalid": int(invalid.get(lvl, 0))})
        print(f"  {lvl:8s} {k / n:.3f}  [{lo:.3f}, {hi:.3f}]   invalid outputs: {invalid.get(lvl, 0)}")
    acc = pd.DataFrame(acc_rows)
    md += ["## 1. Accuracy\n", "| Level | Correct | Accuracy | 95% Wilson CI | Invalid outputs |",
           "|---|---|---|---|---|"]
    md += [f"| {r.level} | {r.correct}/{r.n} | {100 * r.acc:.1f}% | [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}] "
           f"| {r.invalid} |" for r in acc.itertuples()]
    md.append("")

    # ---- 2. omnibus
    if len(levels) > 2 and not is_pilot:
        q = cochrans_q(wide.values)
        k_lv = len(levels)
        eta2 = q.statistic / (n * (k_lv - 1))
        print(f"\n=== 2. Cochran's Q (omnibus) ===\n  Q = {q.statistic:.2f}, df = {q.df}, p = {q.pvalue:.4g}"
              f"\n  effect size eta^2_Q = Q / (n(k-1)) = {eta2:.3f}")
        md += ["## 2. Cochran's Q (omnibus)\n",
               f"- Q({int(q.df)}) = {q.statistic:.2f}, p = {q.pvalue:.3g}",
               f"- Effect size η²_Q = Q / (n(k − 1)) = {eta2:.3f}\n"]

    if args.reference not in levels:
        print(f"\nreference level {args.reference} not in results; skipping pairwise tests")
        return
    others = [l for l in levels if l != args.reference]

    # ---- 3. pairwise
    res = pairwise_vs_reference(wide, args.reference, others)  # power section needs the counts
    if not is_pilot:
        print(f"\n=== 3. Exact McNemar vs {args.reference} (Holm-corrected), with effect sizes ===")
        print_pairwise(res, args.reference)
        md += [f"## 3. Exact McNemar vs {args.reference} (Holm-corrected), with effect sizes\n",
               md_pairwise(res, args.reference)]

    # ---- 4. equivalence (exploratory)
    if args.equivalence_margin and not is_pilot:
        m = args.equivalence_margin
        print(f"\n=== 4. Equivalence check (TOST, margin ±{100 * m:g} pts, alpha 0.05) ===")
        print("  Equivalent if the 90% CI of the difference lies entirely inside the margin.")
        for _, r in res.iterrows():
            inside = -m < r["diff_ci90_lo"] and r["diff_ci90_hi"] < m
            print(f"  {r['level']:8s} 90% CI [{100 * r['diff_ci90_lo']:+.1f}, {100 * r['diff_ci90_hi']:+.1f}]"
                  f"  -> {'equivalent within margin' if inside else 'not shown equivalent'}")
        print("  Report as exploratory unless it was in your analysis plan before data collection.")
        md += [f"## 4. Equivalence check (TOST, margin ±{100 * m:g} pts, α = 0.05)\n",
               "Equivalent if the 90% CI of the difference lies entirely inside the margin. "
               "Report as exploratory unless it was in the analysis plan before data collection.\n",
               "| Comparison | 90% CI (pts) | Result |", "|---|---|---|"]
        md += [f"| {r['level']} vs {args.reference} | [{100 * r['diff_ci90_lo']:+.1f}, "
               f"{100 * r['diff_ci90_hi']:+.1f}] | "
               + ("equivalent within margin" if -m < r["diff_ci90_lo"] and r["diff_ci90_hi"] < m
                  else "not shown equivalent") + " |" for _, r in res.iterrows()]
        md.append("")

    # ---- 5. sensitivity
    print("\n=== 5. Sensitivity analysis: excluding invalid outputs ===")
    bad_items = set(df.loc[~df["valid"], "item_id"]) & set(wide.index)
    md.append("## 5. Sensitivity analysis: excluding invalid outputs\n")
    if not bad_items:
        print("  No invalid outputs at any level; scoring invalid outputs as incorrect had no effect.")
        md.append("No invalid outputs at any level; scoring invalid outputs as incorrect had no effect.\n")
    elif is_pilot:
        print(f"  {len(bad_items)} items had an invalid output at some level. Check the 'raw' field in the "
              "results file\n  and fix the prompt or parsing before the main run.")
        md.append(f"{len(bad_items)} items had an invalid output at some level. Check the `raw` field in the "
                  "results file and fix the prompt or parsing before the main run.\n")
    else:
        sens_wide = wide.drop(index=list(bad_items))
        print(f"  Dropped {len(bad_items)} items with an invalid output at any level (n = {len(sens_wide)}).")
        sens = pairwise_vs_reference(sens_wide, args.reference, others)
        print_pairwise(sens, args.reference)
        changed = (res["p_holm"] < args.alpha) != (sens["p_holm"] < args.alpha)
        md += [f"Dropped {len(bad_items)} items with an invalid output at any level (n = {len(sens_wide)}).\n",
               md_pairwise(sens, args.reference)]
        if changed.any():
            print(f"  ! Conclusions change for: {', '.join(sens.loc[changed, 'level'])}")
            md.append(f"**Conclusions change for: {', '.join(sens.loc[changed, 'level'])}**\n")
        else:
            print("  Same conclusions as the main analysis at alpha = "
                  f"{args.alpha}.")
            md.append(f"Same conclusions as the main analysis at α = {args.alpha}.\n")
        sens.to_csv(out_dir / f"{stem}_sensitivity.csv", index=False)

    # ---- 6. power
    if is_pilot or args.power_analysis:
        md += ["## 6. Power analysis (two-sided McNemar, Connor 1987)\n",
               md_code(tee(power_section, res, n, args, is_pilot))]
    else:
        print("\n=== 6. Power analysis ===\n  Skipped on the main run: power computed from the observed effect "
              "is circular\n  (a non-significant result always looks underpowered). Report the prospective "
              "analysis\n  from the pilot, and use the CIs in section 3 to say what effects the data rule out. "
              "\n  (--power-analysis forces it, e.g. to check the discordance assumption after the run.)")
        md += ["## 6. Power analysis\n",
               "Skipped on the main run: power computed from the observed effect is circular (a non-significant "
               "result always looks underpowered). Report the prospective analysis from the pilot, and use the "
               "CIs in section 3 to say what effects the data rule out.\n"]

    # ---- 7. worked calculations
    if not is_pilot and not args.brief:
        md += ["## 7. Worked calculations\n",
               md_code(tee(worked_calculations, wide, levels, res, args.reference, args.equivalence_margin))]

    # ---- 8/9. outputs
    acc.to_csv(out_dir / f"{stem}_accuracy.csv", index=False)
    report_path = out_dir / f"{stem}_report.md"
    if is_pilot:
        report_path.write_text("\n".join(md))
        print(f"\nwrote {stem}_accuracy.csv, {stem}_report.md to {out_dir}/")
        return
    res.to_csv(out_dir / f"{stem}_pairwise.csv", index=False)
    fig_path = out_dir / f"{stem}_figure.png"
    make_figure(acc, res, args.reference, args.min_effect, fig_path)
    md += ["## Figure\n", f"![Accuracy by quantization level and paired difference vs {args.reference}]"
           f"({fig_path.name})\n"]
    report_path.write_text("\n".join(md))
    print(f"\nwrote {fig_path.name}, {stem}_accuracy.csv, {stem}_pairwise.csv"
          + (f", {stem}_sensitivity.csv" if bad_items else "") + f", {stem}_report.md to {out_dir}/")


if __name__ == "__main__":
    main()