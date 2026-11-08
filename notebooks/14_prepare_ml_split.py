
import os
import pandas as pd

from sklearn.model_selection import train_test_split

# Create output folder
os.makedirs("data/processed/ml", exist_ok=True)

# Load primary dataset
df = pd.read_csv(
    "data/processed/primary_dataset.csv",
    index_col=0
)

# Separate sample IDs and subtype labels
sample_ids = df.index
labels = df["subtype"]

# Stratified 80/20 split
train_ids, test_ids = train_test_split(
    sample_ids,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Create split manifest
manifest = pd.DataFrame({
    "sample_id": sample_ids,
    "subtype": labels.values
})

manifest["split"] = "train"
manifest.loc[
    manifest["sample_id"].isin(test_ids),
    "split"
] = "test"

# Save manifest
manifest.to_csv(
    "data/processed/ml/ml_split_manifest.csv",
    index=False
)

# Display split information
print("ML DATASET SPLIT COMPLETED")
print("--------------------------")

print("Total samples:", len(manifest))
print("Training samples:", (manifest["split"] == "train").sum())
print("Testing samples:", (manifest["split"] == "test").sum())

print("\nTraining subtype distribution:")
print(
    manifest[manifest["split"] == "train"]["subtype"]
    .value_counts()
)

print("\nTesting subtype distribution:")
print(
    manifest[manifest["split"] == "test"]["subtype"]
    .value_counts()
)

print("\nSaved:")
print("data/processed/ml/ml_split_manifest.csv")
