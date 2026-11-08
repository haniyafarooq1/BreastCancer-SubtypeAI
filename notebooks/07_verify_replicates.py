
import pandas as pd

metadata = pd.read_csv("data/processed/sample_metadata.csv")

# Separate original samples and replicates
replicates = metadata[
    metadata["sample_id"].str.endswith("repl")
].copy()

originals = metadata[
    ~metadata["sample_id"].str.endswith("repl")
].copy()

# Remove "repl" to find the original sample ID
replicates["original_id"] = replicates["sample_id"].str.replace(
    "repl", "", regex=False
)

# Match each replicate with its original
check = replicates.merge(
    originals[["sample_id", "subtype"]],
    left_on="original_id",
    right_on="sample_id",
    suffixes=("_replicate", "_original"),
    how="left"
)

print("Original samples:", len(originals))
print("Technical replicates:", len(replicates))
print("Replicates with no matching original:", check["subtype_original"].isna().sum())

# Check subtype agreement
mismatches = check[
    check["subtype_replicate"] != check["subtype_original"]
]

print("Replicates with different subtype labels:", len(mismatches))

if len(mismatches) > 0:
    print("\nMismatches:")
    print(mismatches[[
        "sample_id_replicate",
        "original_id",
        "subtype_replicate",
        "subtype_original"
    ]])

print("\nVerification completed!")
