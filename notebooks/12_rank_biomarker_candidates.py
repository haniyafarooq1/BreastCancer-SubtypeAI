
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

os.makedirs("results/biomarker_candidates", exist_ok=True)

# Load primary dataset
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

X = df.drop(columns=["subtype"]).astype(np.float32)
y = df["subtype"]

# Remove genes with no variation
X = X.loc[:, X.nunique() > 1]

subtypes = sorted(y.unique())

# Compare every possible pair of subtypes
comparisons = list(combinations(subtypes, 2))

for group1, group2 in comparisons:

    print(f"\nComparing {group1} vs {group2}")

    idx1 = np.flatnonzero(y.values == group1)
    idx2 = np.flatnonzero(y.values == group2)

    a = X.iloc[idx1].to_numpy()
    b = X.iloc[idx2].to_numpy()

    results = []
    chunk_size = 1000

    for start in range(0, X.shape[1], chunk_size):

        end = min(start + chunk_size, X.shape[1])

        values1 = a[:, start:end]
        values2 = b[:, start:end]

        U, pvalues = mannwhitneyu(
            values1,
            values2,
            axis=0,
            alternative="two-sided"
        )

        # Rank-biserial effect size
        effect_size = (
            2 * U / (len(idx1) * len(idx2))
        ) - 1

        median_difference = (
            np.median(values1, axis=0)
            - np.median(values2, axis=0)
        )

        genes = X.columns[start:end]

        for gene, p, effect, difference in zip(
            genes, pvalues, effect_size, median_difference
        ):
            results.append(
                (gene, p, effect, difference)
            )

    result_df = pd.DataFrame(
        results,
        columns=[
            "gene",
            "p_value",
            "effect_size",
            "median_difference"
        ]
    )

    result_df["adjusted_p_value"] = multipletests(
        result_df["p_value"],
        method="fdr_bh"
    )[1]

    result_df["significant"] = (
        result_df["adjusted_p_value"] < 0.05
    )

    # Rank by absolute effect size
    result_df["absolute_effect_size"] = (
        result_df["effect_size"].abs()
    )

    result_df = result_df.sort_values(
        "absolute_effect_size",
        ascending=False
    )

    filename = f"{group1}_vs_{group2}.csv"

    result_df.to_csv(
        f"results/biomarker_candidates/{filename}",
        index=False
    )

    print(
        "Significant genes:",
        result_df["significant"].sum()
    )

    print("\nTop 10 genes by effect size:")

    print(
        result_df[
            [
                "gene",
                "effect_size",
                "median_difference",
                "adjusted_p_value"
            ]
        ].head(10).to_string(index=False)
    )

print("\nBIOMARKER CANDIDATE ANALYSIS COMPLETED")
