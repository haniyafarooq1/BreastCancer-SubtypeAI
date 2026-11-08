
import os
import pandas as pd

os.makedirs("results", exist_ok=True)

# Load datasets and results
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

metadata = pd.read_csv(
    "data/processed/sample_metadata.csv"
)

de = pd.read_csv(
    "results/differential_expression/all_gene_results.csv"
)

manifest = pd.read_csv(
    "data/processed/ml/ml_split_manifest.csv"
)

# Prepare report
report = []

report.append("# Breast Cancer Subtype Classification")
report.append("## Data Analysis Summary\n")

report.append("### 1. Dataset")
report.append("- Dataset: GSE96058 (SCAN-B)")
report.append(f"- Primary samples: {len(df)}")
report.append(f"- Gene features: {df.shape[1] - 1}")
report.append(f"- Metadata records: {len(metadata)}")

report.append("\n### 2. Subtype Distribution")
report.append(df["subtype"].value_counts().to_string())

report.append("\n### 3. Data Quality")
report.append(f"- Missing values: {df.isnull().sum().sum()}")
report.append(
    f"- Duplicate sample IDs: {df.index.duplicated().sum()}"
)
report.append(
    f"- Variable genes: {(df.drop(columns=['subtype']).nunique() > 1).sum()}"
)

report.append("\n### 4. Differential Expression Analysis")
report.append("- Statistical test: Kruskal-Wallis")
report.append("- Multiple testing correction: Benjamini-Hochberg FDR")
report.append(f"- Genes tested: {len(de)}")
report.append(
    f"- Significant genes (adjusted p < 0.05): "
    f"{de['significant'].sum()}"
)

report.append("\n### 5. Top Differentially Expressed Genes")
report.append(
    de.head(20)[
        ["gene", "kruskal_statistic", "adjusted_p_value"]
    ].to_string(index=False)
)

report.append("\n### 6. Biomarker Candidate Genes")
report.append(
    "FOXC1, FOXA1, ESR1, GATA3, MLPH, SPDEF, AR, "
    "ABCC11, SOX10, KRT5, KRT14, FOXM1, UBE2C, "
    "BIRC5, AURKA, PTTG1, MYBL2, RRM2, BCL2, XBP1"
)
report.append(
    "\nThese are exploratory candidates and require further validation."
)

report.append("\n### 7. Machine Learning Split")
report.append(
    f"- Training samples: {(manifest['split'] == 'train').sum()}"
)
report.append(
    f"- Testing samples: {(manifest['split'] == 'test').sum()}"
)
report.append("- Split ratio: 80:20")
report.append("- Stratification: By molecular subtype")

report.append("\n### 8. Generated Visualizations")
report.append("- results/eda/subtype_distribution.png")
report.append("- results/eda/expression_distribution.png")
report.append("- results/eda/pca_subtypes.png")
report.append(
    "- results/biomarker_candidates/biomarker_heatmap.png"
)

# Save summary
with open(
    "results/analysis_summary.md",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report))

print("Analysis summary generated successfully!")
print("Saved: results/analysis_summary.md")
