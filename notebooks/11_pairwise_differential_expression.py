
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Create output folder
os.makedirs("results/pairwise_comparisons", exist_ok=True)

# Load primary dataset
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

X = df.drop(columns=["subtype"]).astype(np.float32)
y = df["subtype"]

# Remove genes with no variation
X = X.loc[:, X.nunique() > 1]

# Define subtype comparisons
comparisons = [
    ("LumA", "LumB"),
    ("LumA", "Basal"),
    ("LumA", "Her2"),
    ("LumB", "Basal"),
    ("LumB", "Her2"),
    ("Basal", "Her2"),
]

for group1, group2 in comparisons:

    print(f"\nComparing {group1} vs {group2}")

    idx1 = np.flatnonzero(y.values == group1)
    idx2 = np.flatnonzero(y.values == group2)

    data1 = X.iloc[idx1].to_numpy()
    data2 = X.iloc[idx2].to_numpy()

    results = []
    chunk_size = 1000

    for start in range(0, X.shape[1], chunk_size):

        end = min(start + chunk_size, X.shape[1])

        a = data1[:, start:end]
        b = data2[:, start:end]

        _, pvalues = mannwhitneyu(
            a,
            b,
            axis=0,
            alternative="two-sided"
        )

        genes = X.columns[start:end]

        median_difference = (
            np.median(a, axis=0) -
            np.median(b, axis=0)
        )

        for gene, pvalue, difference in zip(
            genes, pvalues, median_difference
        ):
            results.append((gene, pvalue, difference))

    result_df = pd.DataFrame(
        results,
        columns=["gene", "p_value", "median_difference"]
    )

    result_df["adjusted_p_value"] = multipletests(
        result_df["p_value"],
        method="fdr_bh"
    )[1]

    result_df["significant"] = (
        result_df["adjusted_p_value"] < 0.05
    )

    result_df = result_df.sort_values("adjusted_p_value")

    filename = f"{group1}_vs_{group2}.csv"

    result_df.to_csv(
        f"results/pairwise_comparisons/{filename}",
        index=False
    )

    print(
        "Significant genes:",
        result_df["significant"].sum()
    )

    print("\nTop 10 genes:")
    print(result_df.head(10).to_string(index=False))

print("\nALL PAIRWISE COMPARISONS COMPLETED")
