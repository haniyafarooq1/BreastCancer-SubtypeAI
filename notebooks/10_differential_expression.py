
import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

# Create results folder
os.makedirs("results/differential_expression", exist_ok=True)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/processed/primary_dataset.csv"

df = pd.read_csv(file_path, index_col=0)

# Separate gene expression and subtype labels
X = df.drop(columns=["subtype"]).astype(np.float32)
y = df["subtype"]

# --------------------------------------------------
# 2. Remove genes with no variation
# --------------------------------------------------

X = X.loc[:, X.nunique() > 1]

subtypes = sorted(y.unique())

group_indices = [
    np.flatnonzero(y.values == subtype)
    for subtype in subtypes
]

print("Samples:", len(X))
print("Variable genes:", X.shape[1])
print("Subtypes:", subtypes)

# --------------------------------------------------
# 3. Kruskal-Wallis statistical test
# --------------------------------------------------

print("\nRunning Kruskal-Wallis tests...")

results = []
chunk_size = 1000

for start in range(0, X.shape[1], chunk_size):

    end = min(start + chunk_size, X.shape[1])

    values = X.iloc[:, start:end].to_numpy()

    groups = [
        values[indices, :]
        for indices in group_indices
    ]

    statistic, pvalues = kruskal(*groups, axis=0)

    genes = X.columns[start:end]

    for gene, stat, pvalue in zip(genes, statistic, pvalues):
        results.append((gene, stat, pvalue))

    print(f"Processed {end}/{X.shape[1]} genes")

# --------------------------------------------------
# 4. Create results table
# --------------------------------------------------

de_results = pd.DataFrame(
    results,
    columns=["gene", "kruskal_statistic", "p_value"]
)

# --------------------------------------------------
# 5. Multiple testing correction
# --------------------------------------------------

de_results["adjusted_p_value"] = multipletests(
    de_results["p_value"],
    method="fdr_bh"
)[1]

# Sort genes by adjusted p-value
de_results = de_results.sort_values(
    "adjusted_p_value"
)

# Identify significant genes
de_results["significant"] = (
    de_results["adjusted_p_value"] < 0.05
)

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

de_results.to_csv(
    "results/differential_expression/all_gene_results.csv",
    index=False
)

significant = de_results[
    de_results["significant"]
]

significant.to_csv(
    "results/differential_expression/significant_genes.csv",
    index=False
)

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nANALYSIS COMPLETED")
print("------------------")

print("Genes tested:", len(de_results))

print(
    "Significant genes (adjusted p < 0.05):",
    len(significant)
)

print("\nTop 20 genes:")
print(
    de_results.head(20).to_string(index=False)
)

print("\nResults saved in:")
print("results/differential_expression/")
