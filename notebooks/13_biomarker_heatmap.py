
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results/biomarker_candidates", exist_ok=True)

# Load primary dataset
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

# Candidate genes identified in our analysis
candidate_genes = [
    "FOXC1", "FOXA1", "ESR1", "GATA3", "MLPH",
    "SPDEF", "AR", "ABCC11", "SOX10", "KRT5",
    "KRT14", "FOXM1", "UBE2C", "BIRC5", "AURKA",
    "PTTG1", "MYBL2", "RRM2", "BCL2", "XBP1"
]

# Keep only genes present in the dataset
available_genes = [
    gene for gene in candidate_genes
    if gene in df.columns
]

print("Genes included:", len(available_genes))
print(available_genes)

# Calculate median expression for each subtype
subtype_medians = df.groupby("subtype")[
    available_genes
].median()

# Transpose: genes as rows, subtypes as columns
heatmap_data = subtype_medians.T

# Standardize each gene across subtypes
heatmap_data = heatmap_data.sub(
    heatmap_data.mean(axis=1), axis=0
)

heatmap_data = heatmap_data.div(
    heatmap_data.std(axis=1).replace(0, 1),
    axis=0
)

# Create heatmap
plt.figure(figsize=(10, 11))

sns.heatmap(
    heatmap_data,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    annot=False
)

plt.title("Candidate Gene Expression Across Breast Cancer Subtypes")
plt.xlabel("Molecular Subtype")
plt.ylabel("Candidate Genes")
plt.tight_layout()

plt.savefig(
    "results/biomarker_candidates/biomarker_heatmap.png",
    dpi=300
)

plt.show()

print("\nHeatmap saved successfully!")
