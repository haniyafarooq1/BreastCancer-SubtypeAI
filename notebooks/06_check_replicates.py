
import pandas as pd

metadata = pd.read_csv("data/processed/sample_metadata.csv")

# Find sample IDs containing "repl"
replicates = metadata[
    metadata["sample_id"].str.contains("repl", case=False, na=False)
]

print("Total samples:", len(metadata))
print("Samples marked as replicates:", len(replicates))

print("\nReplicate sample IDs:")
print(replicates[["sample_id", "geo_accession", "subtype"]].to_string(index=False))

# Show examples of all sample IDs
print("\nLast 20 sample IDs:")
print(metadata["sample_id"].tail(20).to_list())
