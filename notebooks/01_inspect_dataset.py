
import pandas as pd

# Path to our downloaded dataset
file_path = "data/raw/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"

print("Reading dataset preview...")

# Read only the first 5 rows to avoid loading the entire file
df = pd.read_csv(file_path, compression="gzip", nrows=5)

print("\nDataset preview:")
print(df.head())

print("\nDataset dimensions (preview):")
print(df.shape)

print("\nFirst 15 column names:")
print(df.columns[:15].tolist())

print("\nFirst 10 row labels:")
print(df.iloc[:10, 0].tolist())
