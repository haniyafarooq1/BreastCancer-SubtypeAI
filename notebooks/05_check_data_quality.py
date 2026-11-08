
import pandas as pd
import numpy as np

# Load combined dataset
df = pd.read_csv("data/processed/combined_dataset.csv", index_col=0)

# Separate gene expression and labels
X = df.drop(columns=["subtype"])
y = df["subtype"]

print("DATASET QUALITY REPORT")
print("-" * 40)

print("Total samples:", X.shape[0])
print("Total genes:", X.shape[1])

# Missing values
print("\nMissing values:", X.isnull().sum().sum())

# Infinite values
print("Infinite values:", np.isinf(X.to_numpy()).sum())

# Duplicate sample IDs
print("Duplicate sample IDs:", df.index.duplicated().sum())

# Duplicate gene names
print("Duplicate gene names:", X.columns.duplicated().sum())

# Genes with no variation
constant_genes = X.columns[X.nunique() <= 1]
print("Genes with no variation:", len(constant_genes))

# Missing subtype labels
print("Missing subtype labels:", y.isnull().sum())

# Expression value range
print("\nMinimum expression value:", X.min().min())
print("Maximum expression value:", X.max().max())

print("\nQuality check completed!")
