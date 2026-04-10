"""
Unified Statistical Analysis for YOLO Epibenthos Detection Study
Version 2.0  (2026-04-10)
================================================================
Supplementary Code for:
  "Implementing Optimized Computer Vision Algorithm To Underwater
   Imagery For Identification and Spatial Analysis Of Epibenthic
   Fauna In Shallow Lagoon Waters"

Authors : Nikolaos Providakis, Georgios Anagnostopoulos, Nikolaos Katsiaras,
          Emanuela Voutsinas, Laura Bray, Ioanna Drakopoulou, Georgia Sarafidou,
          Vasilis Gerakaris, Sofia Reizopoulou
Journal : [Journal]
DOI     : [DOI upon acceptance]

Description
-----------
This script implements the complete statistical methodology described
in the paper's Materials & Methods (Sections S5.1–S5.6) and produces all
publication-ready figures, tables, and supplementary outputs.

Analyses implemented
--------------------
0. Dataset overview & YOLO model performance summary
1. Assumption testing
   - Shapiro-Wilk normality  (overall AND per habitat, per species)
   - Levene homoscedasticity  (between species overall, by habitat, within species)
2. Habitat preference analysis  (Neu et al., 1974; Manly et al., 2002)
   - Area-weighted expected distributions
   - Observed-to-Expected (O/E) preference ratios
   - Significance via standardised residuals (z-scores)
3. Size variation across habitats
   - Global Kruskal-Wallis test
   - Post-hoc pairwise Mann-Whitney U  +  Bonferroni correction  (Dunn, 1961)
4. Habitat complexity correlation
   - Spearman's rank correlation  (ordinal scale; Zar, 2010)
   - Statistical power note for small n
5. Between-species comparative analysis
   - Mann-Whitney U  +  Cohen's d effect size
   - Chi-square contingency for differential habitat use
6. Habitat density table
7. Publication tables (text output, copy-paste ready)
8. Interactive figure generation (12 plot types, user-selectable)
9. Comprehensive Excel export (one workbook, multiple sheets)

Usage
-----
  python unified_statistical_analysis.py

  The script is fully interactive: it prompts for file paths and asks
  which analyses / figures to produce.  Run it in the same directory as
  your data files or supply absolute paths when prompted.

Required input files
--------------------
  Habitat_Epibenthos_statistics.xlsx   (mandatory)
    sheets: 'sizes per habitat Paranem'
            'sizes per hab Anemonia'

  habitat_parameters.txt               (auto-created with defaults on first run)

Python dependencies
-------------------
  pandas >= 1.5
  numpy >= 1.23
  scipy >= 1.9
  matplotlib >= 3.6
  seaborn >= 0.12
  openpyxl >= 3.0   (Excel export)

  Install all with:
    pip install pandas numpy scipy matplotlib seaborn openpyxl

Statistical references cited in the paper
------------------------------------------
  Dunn, O.J. (1961). Multiple comparisons among means.
    J. Am. Stat. Assoc. 56(293), 52–64.
  Manly, B.F.J., McDonald, L.L., Thomas, D.L., McDonald, T.L., &
    Erickson, W.P. (2002). Resource Selection by Animals (2nd ed.).
    Kluwer, Dordrecht.
  Neu, C.W., Byers, C.R., & Peek, J.M. (1974). A technique for
    analysis of utilization-availability data.
    J. Wildl. Manage. 38(3), 541–545.
  Zar, J.H. (2010). Biostatistical Analysis (5th ed.).
    Prentice-Hall, Upper Saddle River, NJ.
  Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram
    Equalization. In Graphics Gems IV (pp. 474–485). Academic Press.

License
-------
  MIT License – see LICENSE file for full text.
  When re-using or adapting this code please cite the original paper.
"""

# ============================================================
#  IMPORTS
# ============================================================
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    kruskal,
    levene,
    mannwhitneyu,
    shapiro,
    spearmanr,
)

warnings.filterwarnings("ignore")

# ── publication-quality plot defaults ──────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.figsize": (14, 10),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    }
)


# ============================================================
#  HELPER UTILITIES
# ============================================================

def _sig_stars(z_or_p, mode="z"):
    """Return significance stars from a z-score or a p-value."""
    if mode == "z":
        v = abs(z_or_p)
        if v > 3.29:
            return "***"
        elif v > 2.58:
            return "**"
        elif v > 1.96:
            return "*"
        return "ns"
    else:  # mode == 'p'
        if z_or_p < 0.001:
            return "***"
        elif z_or_p < 0.01:
            return "**"
        elif z_or_p < 0.05:
            return "*"
        return "ns"


def _oe_label(oe):
    """Interpret an Observed/Expected ratio."""
    if oe >= 1.5:
        return "Strong Pref"
    elif oe >= 1.2:
        return "Preference"
    elif oe >= 0.8:
        return "Neutral"
    elif oe >= 0.5:
        return "Avoidance"
    return "Strong Avoid"


def _cohens_d(a, b):
    """Pooled Cohen's d for two independent samples."""
    n_a, n_b = len(a), len(b)
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (
        n_a + n_b - 2
    )
    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)


def _confirm(prompt="Continue? (y/n) [y]: "):
    """Return True unless the user explicitly types 'n'."""
    return input(prompt).strip().lower() != "n"


def _select_plots(plot_definitions):
    """Interactive plot selection menu; returns list of selected IDs."""
    print("\n" + "─" * 70)
    print("  AVAILABLE FIGURES")
    print("─" * 70)
    for pid, title in plot_definitions.items():
        print(f"  {pid:>2})  {title}")
    print("─" * 70)
    print("  Options:")
    print("    ALL   → generate every figure")
    print("    NONE  → skip figures")
    print("    or enter IDs separated by commas  e.g.  A,C,G")
    print("─" * 70)
    raw = input("  Your choice: ").strip().upper()
    if raw == "ALL":
        return list(plot_definitions.keys())
    if raw in ("NONE", ""):
        return []
    selected = [x.strip() for x in raw.split(",") if x.strip() in plot_definitions]
    if not selected:
        print("  ⚠ No valid IDs recognised – skipping figures.")
    return selected


def _select_analyses(analysis_definitions):
    """Interactive analysis selection menu; returns list of selected keys."""
    print("\n" + "─" * 70)
    print("  SELECT ANALYSES TO RUN")
    print("─" * 70)
    for key, title in analysis_definitions.items():
        print(f"  {key:>2})  {title}")
    print("─" * 70)
    print("  Enter numbers separated by commas, ALL, or press Enter for ALL.")
    raw = input("  Your choice [ALL]: ").strip().upper()
    if raw in ("ALL", ""):
        return list(analysis_definitions.keys())
    selected = [x.strip() for x in raw.split(",") if x.strip() in analysis_definitions]
    return selected if selected else list(analysis_definitions.keys())


# ============================================================
#  MAIN ANALYSIS CLASS
# ============================================================

class UnifiedEpibenthosAnalysis:
    """
    Complete statistical analysis pipeline for YOLO-detected epibenthos.

    Instantiate with the path(s) to your data files; then call
    run_interactive() for the guided menu, or individual methods
    for specific analyses.
    """

    # ── constructor ───────────────────────────────────────────────────────
    def __init__(self, stats_file, output_dir="Plot-results"):
        """
        Initialise the analysis pipeline.

        Parameters
        ----------
        stats_file : str
            Path to the Habitat_Epibenthos_statistics.xlsx workbook.
        output_dir : str
            Directory where generated figures will be saved (created if absent).
        """
        self.stats_file = stats_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._load_dataset()
        self._load_habitat_parameters()
        self._calculate_densities()

    # ── data loading ──────────────────────────────────────────────────────
    def _load_dataset(self):
        """
        Read detection tables from the statistics workbook and combine them.

        Expects two sheets: 'sizes per habitat Paranem' and
        'sizes per hab Anemonia', both containing at least 'Habitat' and
        'diagon_siz' columns.  Rows with any NaN are dropped.
        """
        try:
            self.paranemonia_data = pd.read_excel(
                self.stats_file, sheet_name="sizes per habitat Paranem"
            ).dropna()
            self.anemonia_data = pd.read_excel(
                self.stats_file, sheet_name="sizes per hab Anemonia"
            ).dropna()
        except Exception as exc:
            sys.exit(f"❌  Could not read sheets from {self.stats_file}: {exc}")

        self.paranemonia_data["Species"] = "Paranemonia"
        self.anemonia_data["Species"] = "Anemonia"
        self.combined_data = pd.concat(
            [self.paranemonia_data, self.anemonia_data], ignore_index=True
        )
        print(
            f"✓  {len(self.paranemonia_data):,} Paranemonia  +  "
            f"{len(self.anemonia_data):,} Anemonia  detections loaded"
        )


    def _load_habitat_parameters(self):
        """
        Load habitat area (m²) and structural-complexity ranks from an
        editable text file.  A template is written on first run so that
        reviewers can easily verify or modify the input parameters.
        """
        defaults = {
            "Bare":                                    {"Area_sqm": 26.01, "Complexity_Rank": 1},
            "Mud with Shell Hash":                     {"Area_sqm":  2.92, "Complexity_Rank": 1},
            "Zostera noltei rhizome remnants":         {"Area_sqm":  0.74, "Complexity_Rank": 1},
            "Sparse Macroalgal Thalli":                {"Area_sqm": 11.42, "Complexity_Rank": 2},
            "Patchy/Thin Valonia aegagropila layer":   {"Area_sqm": 13.18, "Complexity_Rank": 3},
            "Dense Valonia aegagropila Bed":           {"Area_sqm": 21.01, "Complexity_Rank": 4},
            "Dense Macroalgal Aggregates":             {"Area_sqm":  8.71, "Complexity_Rank": 5},
        }

        param_file = "habitat_parameters.txt"
        if not os.path.exists(param_file):
            print(f"\n⚠  '{param_file}' not found – creating template with paper default values.")
            with open(param_file, "w") as fh:
                fh.write("Habitat\tArea_sqm\tComplexity_Rank\n")
                for hab, vals in defaults.items():
                    fh.write(f"{hab}\t{vals['Area_sqm']}\t{vals['Complexity_Rank']}\n")
            self.habitat_params = defaults
        else:
            try:
                df_p = pd.read_csv(param_file, sep="\t")
                self.habitat_params = {
                    row["Habitat"]: {
                        "Area_sqm":         float(row["Area_sqm"]),
                        "Complexity_Rank":  int(row["Complexity_Rank"]),
                    }
                    for _, row in df_p.iterrows()
                }
                print(f"✓  Habitat parameters loaded from '{param_file}'")
            except Exception as exc:
                print(f"⚠  Could not read '{param_file}' ({exc}) – using defaults.")
                self.habitat_params = defaults

        print("\n  Habitat".ljust(40) + "Area (m²)".ljust(12) + "Complexity Rank")
        print("  " + "─" * 55)
        for hab, vals in self.habitat_params.items():
            print(f"  {hab[:35]}".ljust(40) + f"{vals['Area_sqm']}".ljust(12) + f"{vals['Complexity_Rank']}")

        if not _confirm("\nAre these habitat parameters correct? (y/n) [y]: "):
            sys.exit(f"Please edit '{param_file}' and re-run the script.")

    def _calculate_densities(self):
        """Compute per-habitat counts and densities; build the core density DataFrame."""
        para_cnt = self.paranemonia_data["Habitat"].value_counts()
        anem_cnt = self.anemonia_data["Habitat"].value_counts()
        rows = []
        for hab, params in self.habitat_params.items():
            area = params["Area_sqm"]
            p_n  = int(para_cnt.get(hab, 0))
            a_n  = int(anem_cnt.get(hab, 0))
            rows.append(
                {
                    "Habitat":             hab,
                    "Area_sqm":            area,
                    "Complexity_Rank":     params["Complexity_Rank"],
                    "Paranemonia_count":   p_n,
                    "Anemonia_count":      a_n,
                    "Total_count":         p_n + a_n,
                    "Paranemonia_density": p_n / area,
                    "Anemonia_density":    a_n / area,
                    "Total_density":       (p_n + a_n) / area,
                }
            )
        self.density_df = pd.DataFrame(rows)


    # ================================================================
    #  SECTION 0  –  DATASET & MODEL SUMMARY
    # ================================================================

    def section0_dataset_summary(self):
        """Print an overview of detection counts, sizes, and YOLO performance."""
        print("\n" + "=" * 80)
        print("  SECTION 0 · DATASET & YOLO MODEL SUMMARY")
        print("=" * 80)

        n_para = len(self.paranemonia_data)
        n_anem = len(self.anemonia_data)
        n_tot  = n_para + n_anem
        print(f"\n  Total individuals detected  : {n_tot:,}")
        print(f"  Paranemonia sp.             : {n_para:,}  ({n_para/n_tot*100:.1f}%)")
        print(f"  Anemonia sp.                : {n_anem:,}  ({n_anem/n_tot*100:.1f}%)")
        print(f"  Species ratio (Para:Anem)   : {n_para/n_anem:.1f} : 1")
        print(f"  Habitat types               : {self.combined_data['Habitat'].nunique()}")

        for sp, df_ in [("Paranemonia", self.paranemonia_data), ("Anemonia", self.anemonia_data)]:
            s = df_["diagon_siz"].describe()
            print(
                f"\n  {sp}  size (m):\n"
                f"    mean ± SD  = {s['mean']:.4f} ± {s['std']:.4f}\n"
                f"    median     = {s['50%']:.4f}\n"
                f"    range      = [{s['min']:.4f}, {s['max']:.4f}]"
            )


        print("\n  Habitat distribution:")
        hab_tbl = (
            self.combined_data.groupby(["Species", "Habitat"])
            .size()
            .unstack(fill_value=0)
        )
        # Custom aligned print: rows = habitats, columns = species
        species_list  = hab_tbl.index.tolist()
        habitat_list  = hab_tbl.columns.tolist()
        hab_w, col_w  = 38, 14
        header = "  " + "Habitat".ljust(hab_w) + "".join(f"{sp:>{col_w}}" for sp in species_list)
        print(header)
        print("  " + "─" * (hab_w + col_w * len(species_list)))
        for hab in habitat_list:
            row = "  " + str(hab)[: hab_w - 2].ljust(hab_w)
            for sp in species_list:
                row += f"{int(hab_tbl.loc[sp, hab]):>{col_w}}"
            print(row)


    # ================================================================
    #  SECTION 1  –  ASSUMPTION TESTING
    # ================================================================

    def section1_assumption_testing(self, detail_level="full"):
        """
        Shapiro-Wilk normality and Levene homoscedasticity tests.

        Parameters
        ----------
        detail_level : 'summary' | 'full'
            'summary' → overall tests only (fast)
            'full'    → also tests per species × habitat (comprehensive)
        """
        print("\n" + "=" * 80)
        print("  SECTION 1 · ASSUMPTION TESTING")
        print("=" * 80)

        # ── 1a. Shapiro-Wilk normality ──────────────────────────────────
        print("\n1a. SHAPIRO-WILK NORMALITY  (Zar, 2010)")
        print("─" * 60)
        norm_rows = []

        def _sw(sizes, label):
            sample = sizes.sample(5000, random_state=42) if len(sizes) > 5000 else sizes
            note   = " (n=5000 subsample)" if len(sizes) > 5000 else ""
            w, p   = shapiro(sample)
            verdict = "Normal" if p > 0.05 else "Non-normal"
            print(f"  {label[:45]}{note}".ljust(60) + f"W={w:.4f}  p={p:.2e}  →  {verdict}")
            norm_rows.append({"Group": label, "n": len(sizes), "W": w, "p": p, "Normal": p > 0.05})

        for sp, df_ in [("Paranemonia", self.paranemonia_data), ("Anemonia", self.anemonia_data)]:
            _sw(df_["diagon_siz"], f"{sp} – Overall")
            if detail_level == "full":
                for hab, grp in df_.groupby("Habitat"):
                    if len(grp) >= 3:
                        _sw(grp["diagon_siz"], f"  {sp} · {hab}")

        self.normality_results = pd.DataFrame(norm_rows)
        n_norm = self.normality_results["Normal"].sum()
        pct    = 100 * n_norm / len(self.normality_results)
        print(f"\n  Normality summary: {n_norm}/{len(self.normality_results)} groups ({pct:.0f}%) passed.")
        print(f"  → {'Non-parametric tests are appropriate (recommended).' if pct < 80 else 'Distributions are approximately normal.'}")

        # ── 1b. Levene homoscedasticity ─────────────────────────────────
        print("\n1b. LEVENE HOMOSCEDASTICITY  (Zar, 2010)")
        print("─" * 60)
        homo_rows = []

        def _lev(groups, label):
            if len(groups) < 2 or any(len(g) < 2 for g in groups):
                return
            stat, p = levene(*groups)
            verdict = "Equal var" if p > 0.05 else "Unequal var"
            print(f"  {label[:55]}".ljust(60) + f"W={stat:.4f}  p={p:.2e}  →  {verdict}")
            homo_rows.append({"Comparison": label, "Levene_W": stat, "p": p, "Equal_var": p > 0.05})

        para_s = self.paranemonia_data["diagon_siz"]
        anem_s = self.anemonia_data["diagon_siz"]
        _lev([para_s, anem_s], "Between species – overall")

        if detail_level == "full":
            common_habs = set(self.paranemonia_data["Habitat"]) & set(self.anemonia_data["Habitat"])
            for hab in sorted(common_habs):
                pg = self.paranemonia_data[self.paranemonia_data["Habitat"] == hab]["diagon_siz"]
                ag = self.anemonia_data[self.anemonia_data["Habitat"] == hab]["diagon_siz"]
                _lev([pg, ag], f"  Between species · {hab}")

            for sp, df_ in [("Paranemonia", self.paranemonia_data), ("Anemonia", self.anemonia_data)]:
                grps = [g["diagon_siz"].values for _, g in df_.groupby("Habitat")]
                _lev(grps, f"  Within {sp} – across habitats")

        self.homosced_results = pd.DataFrame(homo_rows)


    # ================================================================
    #  SECTION 2  –  HABITAT PREFERENCE ANALYSIS
    # ================================================================

    def section2_habitat_preference(self):
        """
        Habitat preference analysis following Neu et al. (1974) and
        Manly et al. (2002).

        Reports:
          - Goodness-of-fit chi-square (area-weighted)
          - O/E ratios and z-score significance per habitat
        """
        print("\n" + "=" * 80)
        print("  SECTION 2 · HABITAT PREFERENCE  (Neu et al., 1974; Manly et al., 2002)")
        print("=" * 80)

        total_area = self.density_df["Area_sqm"].sum()
        area_props = self.density_df["Area_sqm"] / total_area
        self.pref_results = {}

        for sp in ["Paranemonia", "Anemonia"]:
            obs_col = f"{sp}_count"
            obs     = self.density_df[obs_col].to_numpy(dtype=float)
            N       = obs.sum()
            exp     = N * area_props.values

            oe     = np.where(exp > 0, obs / exp, 0.0)
            z      = np.where(exp > 0, (obs - exp) / np.sqrt(exp), 0.0)
            chi2   = float(np.sum((obs - exp) ** 2 / np.where(exp > 0, exp, 1)))
            df_val = len(obs) - 1
            p_gof  = 1 - stats.chi2.cdf(chi2, df_val)

            # Store
            self.pref_results[sp] = {
                "observed": obs, "expected": exp, "oe": oe,
                "z": z, "chi2_gof": chi2, "p_gof": p_gof, "df": df_val,
            }

            # Print table
            print(f"\n  {sp.upper()} SP.")
            print(
                "  Habitat".ljust(37) + "Obs".ljust(6) + "Exp".ljust(8)
                + "O/E".ljust(7) + "z".ljust(9) + "Sig".ljust(6)
                + "Interpretation"
            )
            print("  " + "─" * 80)
            for i, hab in enumerate(self.density_df["Habitat"]):
                print(
                    f"  {hab[:33]}".ljust(37)
                    + f"{int(obs[i])}".ljust(6)
                    + f"{exp[i]:.1f}".ljust(8)
                    + f"{oe[i]:.2f}".ljust(7)
                    + f"{z[i]:+.2f}".ljust(9)
                    + f"{_sig_stars(z[i])}".ljust(6)
                    + _oe_label(oe[i])
                )
            print(
                f"\n  Goodness-of-fit: χ²({df_val}) = {chi2:.2f},  p = {p_gof:.4e}"
            )
            print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")


    # ================================================================
    #  SECTION 3  –  SIZE VARIATION ACROSS HABITATS
    # ================================================================

    def section3_size_variation(self):
        """
        Kruskal-Wallis test for size variation across habitats, followed
        by pairwise Mann-Whitney U tests with Bonferroni correction
        (Dunn, 1961).
        """
        print("\n" + "=" * 80)
        print("  SECTION 3 · SIZE VARIATION ACROSS HABITATS  (Kruskal-Wallis + post-hoc)")
        print("=" * 80)

        self.kw_results = {}

        for sp in ["Paranemonia", "Anemonia"]:
            print(f"\n  {sp.upper()} SP.")
            sp_df = self.combined_data[self.combined_data["Species"] == sp]

            # Descriptive statistics per habitat
            hab_stats = sp_df.groupby("Habitat")["diagon_siz"].agg(
                ["count", "mean", "std", "median", "min", "max"]
            ).round(5)
            print("\n  Descriptive statistics by habitat:")
            print("  " + "─" * 80)
            print(
                "  Habitat".ljust(37) + "n".ljust(7) + "Mean".ljust(9)
                + "SD".ljust(9) + "Median".ljust(9) + "Min".ljust(9) + "Max"
            )
            print("  " + "─" * 80)
            for hab, row in hab_stats.iterrows():
                hab_label = str(hab)[:33]
                print(
                    f"  {hab_label}".ljust(37)
                    + f"{int(row['count'])}".ljust(7)
                    + f"{row['mean']:.4f}".ljust(9)
                    + f"{row['std']:.4f}".ljust(9)
                    + f"{row['median']:.4f}".ljust(9)
                    + f"{row['min']:.4f}".ljust(9)
                    + f"{row['max']:.4f}"
                )

            groups  = {name: grp["diagon_siz"].values for name, grp in sp_df.groupby("Habitat")}
            valid   = {k: v for k, v in groups.items() if len(v) > 0}

            if len(valid) < 3:
                print(f"\n  ⚠  Fewer than 3 habitat groups – Kruskal-Wallis skipped.")
                self.kw_results[sp] = {}
                continue

            H, p_kw = kruskal(*valid.values())
            df_kw   = len(valid) - 1
            print(f"\n  Kruskal-Wallis:  H({df_kw}) = {H:.4f},  p = {p_kw:.4e}")
            print(f"  → {'Significant variation across habitats.' if p_kw < 0.05 else 'No significant variation.'}")

            self.kw_results[sp] = {"H": H, "p": p_kw, "df": df_kw, "pairwise": None}

            if p_kw < 0.05:
                hab_names = list(valid.keys())
                n_comp    = len(list(combinations(hab_names, 2)))
                pairwise  = []
                sig_count = 0

                print(f"\n  Post-hoc pairwise Mann-Whitney U  (Bonferroni, {n_comp} comparisons):")
                print("  " + "─" * 90)
                print(
                    "  Habitat A".ljust(33) + "  Habitat B".ljust(33)
                    + "Adj. p".ljust(12) + "Sig"
                )
                print("  " + "─" * 90)

                for h1, h2 in combinations(hab_names, 2):
                    U, u_p   = mannwhitneyu(valid[h1], valid[h2], alternative="two-sided")
                    adj_p    = min(u_p * n_comp, 1.0)
                    is_sig   = adj_p < 0.05
                    if is_sig:
                        sig_count += 1
                        print(
                            f"  {str(h1)[:30]}".ljust(33)
                            + f"  {str(h2)[:30]}".ljust(33)
                            + f"{adj_p:.4e}".ljust(12)
                            + _sig_stars(adj_p, mode="p")
                        )
                    pairwise.append(
                        {"Habitat_A": h1, "Habitat_B": h2, "U": U,
                         "p_raw": u_p, "p_adj_Bonferroni": adj_p, "Sig": is_sig}
                    )
                print(f"\n  Significant pairs: {sig_count} / {n_comp}")
                self.kw_results[sp]["pairwise"] = pd.DataFrame(pairwise)


    # ================================================================
    #  SECTION 4  –  HABITAT COMPLEXITY CORRELATION
    # ================================================================

    def section4_habitat_complexity(self):
        """
        Spearman rank correlation between species density and habitat
        structural complexity rank  (Zar, 2010).

        Spearman's ρ is used as the primary (and only) test because the
        complexity scale is ordinal and sample size is small (n = number
        of habitat types).  A statistical power note is printed when n < 10.
        """
        print("\n" + "=" * 80)
        print("  SECTION 4 · HABITAT COMPLEXITY CORRELATION  (Spearman; Zar, 2010)")
        print("=" * 80)

        n = len(self.density_df)
        complexity = self.density_df["Complexity_Rank"].values

        self.complexity_results = {}

        for sp in ["Paranemonia", "Anemonia"]:
            dens_col = f"{sp}_density"
            density  = self.density_df[dens_col].values

            rho, p_sp = map(float, spearmanr(complexity, density))

            # Critical r for significance note (two-tailed, α=0.05)
            t_crit = stats.t.ppf(0.975, n - 2)
            r_crit = t_crit / np.sqrt(n - 2 + t_crit**2)

            print(f"\n  {sp.upper()} SP.  (n = {n} habitat types)")
            print(f"  Spearman ρ = {rho:.3f},  p = {p_sp:.4f}")
            print(
                f"  Result: {'Significant' if p_sp < 0.05 else 'Not significant'} "
                f"{'positive' if rho > 0 else 'negative'} monotonic relationship."
            )
            if n < 10:
                print(
                    f"\n  ⚠ Small sample note (n={n} < 10): the critical |ρ| for significance"
                    f" is {r_crit:.3f}.  Treat as exploratory; confidence intervals will be wide."
                )

            self.complexity_results[sp] = {"rho": rho, "p_spearman": p_sp}


    # ================================================================
    #  SECTION 5  –  BETWEEN-SPECIES COMPARATIVE ANALYSIS
    # ================================================================

    def section5_species_comparison(self):
        """
        Between-species comparison:
          - Mann-Whitney U for body size
          - Cohen's d effect size
          - Chi-square test of independence for differential habitat use
        """
        print("\n" + "=" * 80)
        print("  SECTION 5 · SPECIES COMPARATIVE ANALYSIS")
        print("=" * 80)

        para_s = self.paranemonia_data["diagon_siz"]
        anem_s = self.anemonia_data["diagon_siz"]

        # Size comparison
        U, u_p  = mannwhitneyu(para_s, anem_s, alternative="two-sided")
        d       = _cohens_d(para_s.values, anem_s.values)
        d_label = "Large" if abs(d) >= 0.8 else ("Medium" if abs(d) >= 0.5 else "Small")

        print("\n  5a. SIZE COMPARISON  (Mann-Whitney U)")
        print("─" * 55)
        print(f"  Paranemonia  mean ± SD = {para_s.mean():.4f} ± {para_s.std():.4f} m  (n={len(para_s):,})")
        print(f"  Anemonia     mean ± SD = {anem_s.mean():.4f} ± {anem_s.std():.4f} m  (n={len(anem_s):,})")
        print(f"  Mann-Whitney U = {U:.0f},  p = {u_p:.2e}  {_sig_stars(u_p, mode='p')}")
        print(f"  Cohen's d = {d:.3f}  ({d_label} effect; Anemonia {abs(d):.1f} SD {'larger' if d < 0 else 'smaller'})")

        # Habitat use comparison (chi-square contingency)
        contingency = pd.crosstab(self.combined_data["Species"], self.combined_data["Habitat"])
        chi2_stat, chi2_p, dof, _ = chi2_contingency(contingency.values)
        chi2 = chi2_stat
        chi2_p = float(chi2_p)

        print("\n  5b. DIFFERENTIAL HABITAT USE  (Chi-square contingency)")
        print("─" * 55)
        print(f"  χ²({dof}) = {chi2:.2f},  p = {chi2_p:.2e}  {_sig_stars(chi2_p, mode='p')}")
        print(
            f"  → {'Significant' if chi2_p < 0.05 else 'No significant'} "
            "difference in habitat use between species."
        )
        print("\n  Proportional habitat use (% of each species' individuals):")
        pct_tbl = contingency.div(contingency.sum(axis=1), axis=0).mul(100).round(1)
        # Custom aligned print: rows = habitats, columns = species
        species_list  = pct_tbl.index.tolist()
        habitat_list  = pct_tbl.columns.tolist()
        hab_w, col_w  = 38, 14
        print("  " + "Habitat".ljust(hab_w) + "".join(f"{sp:>{col_w}}" for sp in species_list))
        print("  " + "─" * (hab_w + col_w * len(species_list)))
        for hab in habitat_list:
            row = "  " + str(hab)[: hab_w - 2].ljust(hab_w)
            for sp in species_list:
                row += f"{pct_tbl.loc[sp, hab]:>{col_w}.1f}"
            print(row)

        self.species_comparison = {
            "U": U, "p_mw": u_p, "cohens_d": d,
            "chi2": chi2, "p_chi2": chi2_p, "dof": dof,
        }


    # ================================================================
    #  SECTION 6  –  DENSITY TABLE
    # ================================================================

    def section6_density_table(self):
        """Print the full per-habitat density table and inter-species density correlation."""
        print("\n" + "=" * 80)
        print("  SECTION 6 · HABITAT DENSITY TABLE")
        print("=" * 80)

        print(
            "\n  Habitat".ljust(37) + "Area(m²)".ljust(10)
            + "Para cnt".ljust(10) + "Anem cnt".ljust(10)
            + "Para d".ljust(10) + "Anem d".ljust(10) + "Total d"
        )
        print("  " + "─" * 90)
        for _, r in self.density_df.iterrows():
            print(
                f"  {r['Habitat'][:33]}".ljust(37)
                + f"{r['Area_sqm']:.1f}".ljust(10)
                + f"{int(r['Paranemonia_count'])}".ljust(10)
                + f"{int(r['Anemonia_count'])}".ljust(10)
                + f"{r['Paranemonia_density']:.2f}".ljust(10)
                + f"{r['Anemonia_density']:.2f}".ljust(10)
                + f"{r['Total_density']:.2f}"
            )
        print("  (d = density in ind/m²)")


    # ================================================================
    #  SECTION 7  –  TABLES IN PUBLICATION RESULTS
    # ================================================================

    def section7_publication_tables(self):
        """Print copy-paste-ready publication tables (Table 1, 2, 3)."""
        print("\n" + "=" * 80)
        print("  SECTION 7 · TABLES IN PUBLICATION RESULTS")
        print("=" * 80)

        # Table 1: Summary statistics
        print("\n  TABLE 1.  Summary statistics for YOLO-detected epibenthos")
        print("─" * 90)
        tbl1 = (
            self.combined_data.groupby(["Species", "Habitat"])["diagon_siz"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .round(4)
        )
        tbl1.columns = ["n", "Mean (m)", "SD (m)", "Median (m)", "Min (m)", "Max (m)"]
        print(tbl1.to_string())

        # Table 2: Preference ratios
        if self.pref_results:
            print("\n  TABLE 2.  Habitat preference analysis (Neu et al., 1974)")
            print(
                "  Habitat".ljust(38)
                + "Para O/E".ljust(11) + "Para Sig".ljust(10)
                + "Anem O/E".ljust(11) + "Anem Sig"
            )
            print("  " + "─" * 75)
            for i, hab in enumerate(self.density_df["Habitat"]):
                pr = self.pref_results["Paranemonia"]
                ar = self.pref_results["Anemonia"]
                print(
                    f"  {hab[:34]}".ljust(38)
                    + f"{pr['oe'][i]:.2f}".ljust(11)
                    + f"{_sig_stars(pr['z'][i])}".ljust(10)
                    + f"{ar['oe'][i]:.2f}".ljust(11)
                    + f"{_sig_stars(ar['z'][i])}"
                )
            print("  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
            print("  O/E = Observed/Expected ratio")

        # Table 3: Statistical tests summary
        print("\n  TABLE 3.  Summary of statistical tests")
        print("─" * 80)
        rows3 = []
        if hasattr(self, "species_comparison"):
            sc = self.species_comparison
            rows3.append(("Mann-Whitney U (size, between species)",
                          f"U = {sc['U']:.0f}", f"{sc['p_mw']:.2e}",
                          _sig_stars(sc['p_mw'], mode='p')))
            rows3.append(("Chi-square (habitat use, between species)",
                          f"χ²({sc['dof']}) = {sc['chi2']:.2f}", f"{sc['p_chi2']:.2e}",
                          _sig_stars(sc['p_chi2'], mode='p')))
        for sp, res in self.kw_results.items():
            if "H" in res:
                rows3.append((f"Kruskal-Wallis ({sp} size by habitat)",
                              f"H({res['df']}) = {res['H']:.2f}", f"{res['p']:.4f}",
                              _sig_stars(res['p'], mode='p')))
        for sp, res in self.complexity_results.items():
            rows3.append((f"Spearman ρ ({sp} density vs complexity)",
                          f"ρ = {res['rho']:.3f}", f"{res['p_spearman']:.4f}",
                          _sig_stars(res['p_spearman'], mode='p')))
        for label, stat, p_val, sig in rows3:
            print(f"  {label[:52]}".ljust(55) + f"{stat:<18}{p_val:<12}{sig}")


    # ================================================================
    #  SECTION 8  –  FIGURES
    # ================================================================

    def section8_figures(self, selected=None):
        """
        Produce publication-quality figures.

        Parameters
        ----------
        selected : list of str, or None
            IDs of plots to generate.  If None, the user is prompted.
        """
        plot_defs = {
            "A":  "Species body-size comparison (boxplot)",
            "B":  "Paranemonia habitat distribution (pie chart)",
            "C":  "Anemonia habitat distribution (pie chart)",
            "D":  "Paranemonia body-size by habitat (boxplot)",
            "E":  "Anemonia body-size by habitat (boxplot)",
            "F":  "Size-distribution overlap (histogram)",
            "G":  "Proportional habitat use (bar chart)",
            "H":  "O/E preference ratios (bar chart)",
            "I":  "Habitat density comparison (bar chart)",
            "J":  "Density vs. complexity (scatter + Spearman ρ)",
            "K":  "Preference summary heatmap",
        }

        if selected is None:
            selected = _select_plots(plot_defs)
        if not selected:
            print("  ↳ No figures selected – skipping.")
            return

        print(f"\n  Generating {len(selected)} figure(s) → '{self.output_dir}/'")

        for pid in selected:
            fig, ax = plt.subplots(figsize=(12, 7))
            title   = f"{pid}) {plot_defs.get(pid, '')}"

            if pid == "A":
                sns.boxplot(data=self.combined_data, x="Species", y="diagon_siz", ax=ax)
                ax.set_ylabel("Diagonal size (m)")
                p_m  = self.paranemonia_data["diagon_siz"].mean()
                a_m  = self.anemonia_data["diagon_siz"].mean()
                ax.text(0.5, 0.97, f"Size ratio  Anem:Para = {a_m/p_m:.1f}×",
                        transform=ax.transAxes, ha="center", va="top",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

            elif pid == "B":
                cnts = self.paranemonia_data["Habitat"].value_counts()
                ax.pie(
                    cnts.to_numpy(),
                    labels=cnts.index.astype(str).tolist(),
                    autopct="%1.1f%%",
                    colors=sns.color_palette("husl", len(cnts)),
                    startangle=90,
                )

            elif pid == "C":
                cnts = self.anemonia_data["Habitat"].value_counts()
                ax.pie(
                    cnts.to_numpy(),
                    labels=cnts.index.astype(str).tolist(),
                    autopct="%1.1f%%",
                    colors=sns.color_palette("husl", len(cnts)),
                    startangle=90,
                )

            elif pid == "D":
                sns.boxplot(
                    data=self.combined_data[self.combined_data["Species"] == "Paranemonia"],
                    x="Habitat", y="diagon_siz", ax=ax,
                )
                ax.set_ylabel("Diagonal size (m)")
                ax.tick_params(axis="x", rotation=40)

            elif pid == "E":
                sns.boxplot(
                    data=self.combined_data[self.combined_data["Species"] == "Anemonia"],
                    x="Habitat", y="diagon_siz", ax=ax,
                )
                ax.set_ylabel("Diagonal size (m)")
                ax.tick_params(axis="x", rotation=40)

            elif pid == "F":
                p_s = self.paranemonia_data["diagon_siz"]
                a_s = self.anemonia_data["diagon_siz"]
                ax.hist(p_s, bins=50, alpha=0.7, density=True, label="Paranemonia", color="steelblue")
                ax.hist(a_s, bins=30, alpha=0.7, density=True, label="Anemonia",    color="tomato")
                ax.set_xlabel("Diagonal size (m)")
                ax.set_ylabel("Density")
                ax.legend()

            elif pid == "G":
                hab_pct = (
                    pd.crosstab(self.combined_data["Habitat"], self.combined_data["Species"])
                    .div(self.combined_data.groupby("Species").size(), axis=1)
                    .mul(100)
                )
                hab_pct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], alpha=0.85)
                ax.set_ylabel("% of species total")
                ax.set_xlabel("Habitat type")
                ax.tick_params(axis="x", rotation=40)
                ax.legend(title="Species")

            elif pid == "H":
                if self.pref_results:
                    habs  = self.density_df["Habitat"].tolist()
                    x     = np.arange(len(habs))
                    w     = 0.35
                    p_oe  = self.pref_results["Paranemonia"]["oe"]
                    a_oe  = self.pref_results["Anemonia"]["oe"]
                    ax.bar(x - w/2, p_oe, w, label="Paranemonia", alpha=0.85, color="steelblue")
                    ax.bar(x + w/2, a_oe, w, label="Anemonia",    alpha=0.85, color="tomato")
                    ax.axhline(1.0, color="black",  ls="-",  lw=1.2, alpha=0.6, label="Random (1.0)")
                    ax.axhline(1.5, color="green",  ls="--", lw=1.0, alpha=0.7, label="Strong pref (1.5)")
                    ax.axhline(0.5, color="red",    ls="--", lw=1.0, alpha=0.7, label="Strong avoid (0.5)")
                    ax.set_xticks(x)
                    ax.set_xticklabels(habs, rotation=40, ha="right")
                    ax.set_ylabel("Observed / Expected ratio")
                    ax.legend(fontsize=9)

            elif pid == "I":
                x     = np.arange(len(self.density_df))
                w     = 0.35
                habs  = self.density_df["Habitat"].tolist()
                ax.bar(x - w/2, self.density_df["Paranemonia_density"], w,
                       label="Paranemonia", alpha=0.85, color="steelblue")
                ax.bar(x + w/2, self.density_df["Anemonia_density"], w,
                       label="Anemonia",    alpha=0.85, color="tomato")
                ax.set_xticks(x)
                ax.set_xticklabels(habs, rotation=40, ha="right")
                ax.set_ylabel("Density  (ind m⁻²)")
                ax.legend()

            elif pid == "J":
                fig.set_size_inches(10, 7)
                for sp, col, mrk in [("Paranemonia", "steelblue", "o"), ("Anemonia", "tomato", "s")]:
                    dens = self.density_df[f"{sp}_density"].to_numpy(dtype=float)
                    comp = self.density_df["Complexity_Rank"].to_numpy(dtype=float)
                    idx_max = int(np.argmax(dens))
                    ax.scatter(comp, dens, color=col, marker=mrk, s=80, label=sp, zorder=3)
                    # regression line
                    m, b  = np.polyfit(comp, dens, 1)
                    x_fit = np.linspace(comp.min(), comp.max(), 100)
                    ax.plot(x_fit, m * x_fit + b, color=col, lw=1.5, ls="--", alpha=0.7)
                    rho, p = spearmanr(comp, dens)
                    ax.annotate(
                        f"{sp}\nρ={rho:.2f}, p={p:.3f}",
                        xy=(float(comp[idx_max]), float(np.max(dens))),
                        xytext=(5, 5), textcoords="offset points", fontsize=9, color=col,
                    )
                ax.set_xlabel("Habitat structural complexity rank")
                ax.set_ylabel("Species density  (ind m⁻²)")
                ax.legend()

            elif pid == "K":
                if self.pref_results:
                    habs = self.density_df["Habitat"].tolist()
                    # encode: +2 strong pref, +1 pref, 0 neutral, -1 avoid, -2 strong avoid
                    def _encode(oe_arr):
                        out = []
                        for v in oe_arr:
                            if   v >= 1.5: out.append(2)
                            elif v >= 1.2: out.append(1)
                            elif v >= 0.8: out.append(0)
                            elif v >= 0.5: out.append(-1)
                            else:          out.append(-2)
                        return out
                    matrix = np.array([
                        _encode(self.pref_results["Paranemonia"]["oe"]),
                        _encode(self.pref_results["Anemonia"]["oe"]),
                    ])
                    sns.heatmap(
                        matrix, annot=True, fmt="d", cmap="RdBu_r", center=0,
                        xticklabels=habs, yticklabels=["Paranemonia", "Anemonia"],
                        cbar_kws={"label": "Preference score (−2 to +2)"},
                        ax=ax,
                    )
                    ax.tick_params(axis="x", rotation=40)

            ax.set_title(title, fontweight="bold")
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, f"fig_{pid}.png")
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓  {out_path}")

        # Optional combined panel
        if len(selected) > 1:
            if _confirm("\nSave selected figures in a combined panel image? (y/n) [y]: "):
                cols = 3
                rows = int(np.ceil(len(selected) / cols))
                fig_c, axes_c = plt.subplots(rows, cols, figsize=(18, 6 * rows))
                fig_c.suptitle("YOLO Epibenthos – Statistical Analysis", fontsize=15, fontweight="bold")
                axes_flat = axes_c.flatten()
                for idx, pid in enumerate(selected):
                    _draw_on_ax(self, pid, axes_flat[idx], plot_defs[pid])
                for ax_ in axes_flat[len(selected):]:
                    ax_.set_visible(False)
                plt.tight_layout()
                plt.subplots_adjust(top=0.94)
                combined_path = os.path.join(self.output_dir, "combined_panel.png")
                fig_c.savefig(combined_path, dpi=500, bbox_inches="tight")
                plt.close(fig_c)
                print(f"  ✓  Combined panel → {combined_path}")


    # ================================================================
    #  SECTION 9  –  EXCEL EXPORT
    # ================================================================

    def section9_export_excel(self, filename=None):
        """
        Export all results to a comprehensive Excel workbook
        (one workbook, multiple sheets).
        """
        if filename is None:
            filename = input(
                "\n  Export filename (default: epibenthos_results.xlsx): "
            ).strip() or "epibenthos_results.xlsx"

        print(f"\n  💾  Exporting to '{filename}' …")
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:

            # Raw data
            self.paranemonia_data.to_excel(writer, sheet_name="Paranemonia_raw", index=False)
            self.anemonia_data.to_excel(writer,    sheet_name="Anemonia_raw",    index=False)
            self.combined_data.to_excel(writer,    sheet_name="Combined_data",   index=False)

            # Density table
            self.density_df.to_excel(writer, sheet_name="Density_table", index=False)

            # Summary statistics
            summ = (
                self.combined_data.groupby(["Species", "Habitat"])["diagon_siz"]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(5)
            )
            summ.to_excel(writer, sheet_name="Summary_statistics")

            # Normality tests
            if hasattr(self, "normality_results"):
                self.normality_results.to_excel(writer, sheet_name="Normality_tests", index=False)

            # Homoscedasticity tests
            if hasattr(self, "homosced_results"):
                self.homosced_results.to_excel(writer, sheet_name="Homoscedasticity", index=False)

            # Preference results
            if self.pref_results:
                for sp in ["Paranemonia", "Anemonia"]:
                    pr  = self.pref_results[sp]
                    df_ = pd.DataFrame({
                        "Habitat":   self.density_df["Habitat"].tolist(),
                        "Observed":  pr["observed"].astype(int),
                        "Expected":  np.round(pr["expected"], 2),
                        "O_over_E":  np.round(pr["oe"], 3),
                        "z_score":   np.round(pr["z"], 3),
                        "Sig":       [_sig_stars(z) for z in pr["z"]],
                    })
                    df_.to_excel(writer, sheet_name=f"Pref_{sp[:11]}", index=False)

            # Kruskal-Wallis
            for sp, res in self.kw_results.items():
                if res.get("pairwise") is not None:
                    res["pairwise"].to_excel(
                        writer, sheet_name=f"KW_pairwise_{sp[:8]}", index=False
                    )

        print(f"  ✓  Saved {filename}")


    # ================================================================
    #  INTERACTIVE RUNNER
    # ================================================================

    def run_interactive(self):
        """
        Guided interactive menu: select which analyses and figures to produce.
        Run this after instantiation, or call individual section_* methods directly.
        """
        analysis_defs = {
            "0": "Dataset & model summary",
            "1": "Assumption testing (normality + homoscedasticity)",
            "2": "Habitat preference analysis  (Neu 1974; O/E ratios + z-scores)",
            "3": "Size variation across habitats  (Kruskal-Wallis + Bonferroni post-hoc)",
            "4": "Habitat complexity correlation  (Spearman ρ)",
            "5": "Between-species comparison  (Mann-Whitney U, Cohen's d, chi-square)",
            "6": "Habitat density table",
            "7": "Publication tables",
            "8": "Figures (interactive figure selection)",
            "9": "Excel export",
        }

        chosen = _select_analyses(analysis_defs)

        if "0"  in chosen: self.section0_dataset_summary()

        detail = "full"
        if "1" in chosen:
            d = input("\n  Normality / homoscedasticity detail level: [F]ull by-habitat or [S]ummary only? [F]: ").strip().upper()
            detail = "summary" if d == "S" else "full"
            self.section1_assumption_testing(detail_level=detail)

        if "2"  in chosen: self.section2_habitat_preference()
        if "3"  in chosen: self.section3_size_variation()
        if "4"  in chosen: self.section4_habitat_complexity()
        if "5"  in chosen: self.section5_species_comparison()
        if "6"  in chosen: self.section6_density_table()
        if "7"  in chosen: self.section7_publication_tables()
        if "8"  in chosen: self.section8_figures()
        if "9"  in chosen: self.section9_export_excel()

        print("\n" + "=" * 80)
        print("  ✅  All selected analyses completed.")
        print("=" * 80)


# ── helper for combined panel (needs access to self) ─────────────────────────
def _draw_on_ax(obj, pid, ax, title):
    """Minimal re-draw of a plot into a pre-existing Axes (used by combined panel)."""
    ax.set_title(f"{pid}) {title[:45]}", fontweight="bold", fontsize=10)

    if pid == "A":
        sns.boxplot(data=obj.combined_data, x="Species", y="diagon_siz", ax=ax)
        ax.set_ylabel("Size (m)")
    elif pid in ("B", "C"):
        sp_   = "Paranemonia" if pid == "B" else "Anemonia"
        df_   = obj.paranemonia_data if pid == "B" else obj.anemonia_data
        cnts  = df_["Habitat"].value_counts()
        ax.pie(cnts.to_numpy(), labels=None, autopct="%1.0f%%",
               colors=sns.color_palette("husl", len(cnts)), startangle=90)
    elif pid == "D":
        sns.boxplot(
            data=obj.combined_data[obj.combined_data["Species"] == "Paranemonia"],
            x="Habitat", y="diagon_siz", ax=ax)
        ax.set_ylabel("Size (m)")
        ax.tick_params(axis="x", rotation=45)
    elif pid == "E":
        sns.boxplot(
            data=obj.combined_data[obj.combined_data["Species"] == "Anemonia"],
            x="Habitat", y="diagon_siz", ax=ax)
        ax.set_ylabel("Size (m)")
        ax.tick_params(axis="x", rotation=45)
    elif pid == "F":
        p_s = obj.paranemonia_data["diagon_siz"]
        a_s = obj.anemonia_data["diagon_siz"]
        ax.hist(p_s, bins=50, alpha=0.7, density=True, label="Paranemonia", color="steelblue")
        ax.hist(a_s, bins=30, alpha=0.7, density=True, label="Anemonia",    color="tomato")
        ax.set_xlabel("Size (m)")
        ax.legend(fontsize=8)
    elif pid == "G":
        hab_pct = (
            pd.crosstab(obj.combined_data["Habitat"], obj.combined_data["Species"])
            .div(obj.combined_data.groupby("Species").size(), axis=1).mul(100)
        )
        hab_pct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], alpha=0.85, legend=True)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("%")
    elif pid == "I" and obj.pref_results:
        x = np.arange(len(obj.density_df))
        w = 0.35
        ax.bar(x - w/2, obj.pref_results["Paranemonia"]["oe"], w, color="steelblue", alpha=0.85, label="Paranemonia")
        ax.bar(x + w/2, obj.pref_results["Anemonia"]["oe"],    w, color="tomato",    alpha=0.85, label="Anemonia")
        ax.axhline(1, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(obj.density_df["Habitat"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("O/E")
        ax.legend(fontsize=8)
    elif pid == "K":
        x = np.arange(len(obj.density_df))
        w = 0.35
        ax.bar(x - w/2, obj.density_df["Paranemonia_density"], w, color="steelblue", alpha=0.85, label="Paranemonia")
        ax.bar(x + w/2, obj.density_df["Anemonia_density"],    w, color="tomato",    alpha=0.85, label="Anemonia")
        ax.set_xticks(x)
        ax.set_xticklabels(obj.density_df["Habitat"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ind m⁻²")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Plot {pid}\n(see individual file)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)


# ============================================================
#  OUTPUT CAPTURE (for Markdown export)
# ============================================================

import io as _io

class _Tee:
    """Mirror every print() to both the real stdout and an in-memory buffer."""
    def __init__(self, real_stdout):
        self._real = real_stdout
        self._buf  = _io.StringIO()
    def write(self, data):
        self._real.write(data)
        self._buf.write(data)
    def flush(self):
        self._real.flush()
    def getvalue(self):
        return self._buf.getvalue()
    def isatty(self):
        return getattr(self._real, "isatty", lambda: False)()


# ============================================================
#  ENTRY POINT
# ============================================================

def _clean_path(raw: str) -> str:
    """Strip surrounding single/double quotes and whitespace from a user-typed path."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
        raw = raw[1:-1].strip()
    return raw


def main():
    """
    Command-line entry point.

    Prompts the user for file paths and an output directory, constructs
    a UnifiedEpibenthosAnalysis instance, and launches the interactive
    analysis menu.  All prompts have sensible defaults so the script
    can be run without any command-line arguments.
    """
    print("=" * 80)
    print("  YOLO Epibenthos Detection – Unified Statistical Analysis")
    print("  Supplementary code for [Implementing Optimized Computer "
          "Vision Algorithm To Underwater Imagery For Identification "
          "and Spatial Analysis Of Epibenthic Fauna In Shallow Lagoon Waters / DOI]")
    print("=" * 80)

    # ── file paths ──────────────────────────────────────────────────────
    print("\n  INPUT FILES")
    print("─" * 55)

    stats_file = _clean_path(input(
        "  Path to Habitat_Epibenthos statistics file\n"
        "  (e.g. Habitat_Epibenthos_statistics.xlsx): "
    ))

    if not os.path.isfile(stats_file):
        sys.exit(f"\n❌  File not found: '{stats_file}'  –  please check the path and retry.")


    out_dir = _clean_path(input(
        "\n  Output directory for figures (default: Plot-results): "
    )) or "Plot-results"

    # ── initialise and run ──────────────────────────────────────────────
    analysis = UnifiedEpibenthosAnalysis(stats_file, output_dir=out_dir)

    # Capture all print() output for optional Markdown export
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    try:
        analysis.run_interactive()
    finally:
        sys.stdout = tee._real

    # ── optional Markdown export ────────────────────────────────────────
    md_raw = input(
        "\n  Export console log as Markdown? Enter filename (or press Enter to skip): "
    ).strip()
    if md_raw:
        md_path = _clean_path(md_raw)
        if not md_path.lower().endswith(".md"):
            md_path += ".md"
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# YOLO Epibenthos – Unified Statistical Analysis\n\n")
            fh.write("```text\n")
            fh.write(tee.getvalue())
            fh.write("\n```\n")
        print(f"  ✓  Console log saved → {md_path}")


if __name__ == "__main__":
    main()
