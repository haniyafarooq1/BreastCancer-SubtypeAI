
import pandas as pd

# Load gene expression data
expression = pd.read_csv(
    "data/raw/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz",
    compression="gzip",
    index_col=0
)

# Convert: rows = samples, columns = genes
expression = expression.T
expression.index.name = "sample_id"

# Load sample metadata
metadata = pd.read_csv("data/processed/sample_metadata.csv")

metadata = metadata.set_index("sample_id")

# Check whether every expression sample has metadata
missing_samples = expression.index.difference(metadata.index)
extra_samples = metadata.index.difference(expression.index)

print("Expression samples:", len(expression))
print("Metadata samples:", len(metadata))
print("Samples missing metadata:", len(missing_samples))
print("Metadata samples missing expression:", len(extra_samples))

# Match metadata to expression samples
dataset = expression.join(metadata[["subtype"]], how="inner")

# Save combined dataset
dataset.to_csv("data/processed/combined_dataset.csv")

print("\nCombined dataset created!")
print("Dataset shape:", dataset.shape)

print("\nSubtype distribution:")
print(dataset["subtype"].value_counts())

print("\nFirst 5 rows and 5 columns:")
print(dataset.iloc[:5, :5])
