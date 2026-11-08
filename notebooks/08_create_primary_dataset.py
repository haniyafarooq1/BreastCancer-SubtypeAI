
import pandas as pd

# Load combined dataset
df = pd.read_csv(
    "data/processed/combined_dataset.csv",
    index_col=0
)

# Separate technical replicates
replicate_mask = df.index.str.endswith("repl")

replicates = df[replicate_mask].copy()
primary = df[~replicate_mask].copy()

# Save both datasets
primary.to_csv("data/processed/primary_dataset.csv")
replicates.to_csv("data/processed/technical_replicates.csv")

print("PRIMARY DATASET CREATED")
print("-----------------------")

print("Original samples:", len(primary))
print("Technical replicates:", len(replicates))
print("Total samples:", len(df))

print("\nPrimary dataset shape:", primary.shape)

print("\nSubtype distribution:")
print(primary["subtype"].value_counts())

print("\nSaved:")
print("data/processed/primary_dataset.csv")
print("data/processed/technical_replicates.csv")
