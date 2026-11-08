
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create results folder
os.makedirs("results/eda", exist_ok=True)

# Load primary dataset
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

X = df.drop(columns=["subtype"])
y = df["subtype"]

print("Dataset:", X.shape)
print("\nSubtype counts:")
print(y.value_counts())

# --------------------------------------------------
# 1. Subtype distribution
# --------------------------------------------------

counts = y.value_counts()

plt.figure(figsize=(9, 5))
counts.plot(kind="bar")
plt.title("Distribution of Breast Cancer Molecular Subtypes")
plt.xlabel("Molecular Subtype")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/eda/subtype_distribution.png", dpi=300)
plt.close()

# --------------------------------------------------
# 2. Gene expression value distribution
# --------------------------------------------------

# Randomly select 500 genes and 500 samples
rng = np.random.default_rng(42)

selected_genes = rng.choice(
    X.columns,
    size=min(500, X.shape[1]),
    replace=False
)

selected_samples = X.sample(
    n=min(500, len(X)),
    random_state=42
)

values = selected_samples[selected_genes].to_numpy().flatten()

plt.figure(figsize=(9, 5))
plt.hist(values, bins=60)
plt.title("Distribution of Gene Expression Values")
plt.xlabel("Expression Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/eda/expression_distribution.png", dpi=300)
plt.close()

# --------------------------------------------------
# 3. PCA visualization
# --------------------------------------------------

print("\nRunning PCA...")

# Select the 2000 most variable genes
gene_variances = X.var()
top_genes = gene_variances.nlargest(2000).index

X_selected = X[top_genes]

# PCA reduces thousands of genes to 2 dimensions
pca = PCA(n_components=2, svd_solver="randomized", random_state=42)

pca_result = pca.fit_transform(X_selected)

pca_df = pd.DataFrame(
    pca_result,
    columns=["PC1", "PC2"],
    index=X.index
)

pca_df["subtype"] = y

# Plot samples by subtype
plt.figure(figsize=(10, 7))

for subtype in sorted(y.unique()):
    subset = pca_df[pca_df["subtype"] == subtype]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=subtype,
        alpha=0.6,
        s=18
    )

plt.title("PCA of Breast Cancer Gene Expression Data")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig("results/eda/pca_subtypes.png", dpi=300)
plt.close()

# Save PCA coordinates
pca_df.to_csv("results/eda/pca_coordinates.csv")

# Save the selected genes
pd.Series(top_genes, name="gene").to_csv(
    "results/eda/top_2000_variable_genes.csv",
    index=False
)

print("\nPCA variance explained:")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")
print(
    "Total:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print("\nEDA completed!")
print("Graphs saved in results/eda/")
