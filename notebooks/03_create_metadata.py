
import gzip
import csv
import pandas as pd

files = [
    "data/raw/GSE96058-GPL11154_series_matrix.txt.gz",
    "data/raw/GSE96058-GPL18573_series_matrix.txt.gz"
]

all_samples = []

for file_path in files:
    accessions = []
    titles = []
    subtypes = []

    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                accessions = next(csv.reader([line], delimiter="\t"))[1:]

            elif line.startswith("!Sample_title"):
                titles = next(csv.reader([line], delimiter="\t"))[1:]

            elif line.startswith("!Sample_characteristics_ch1"):
                values = next(csv.reader([line], delimiter="\t"))[1:]

                if values and "pam50 subtype:" in values[0].lower():
                    subtypes = [
                        value.split(":", 1)[1].strip()
                        for value in values
                    ]

    for accession, title, subtype in zip(accessions, titles, subtypes):
        all_samples.append({
            "sample_id": title,
            "geo_accession": accession,
            "subtype": subtype
        })

metadata = pd.DataFrame(all_samples)

metadata.to_csv(
    "data/processed/sample_metadata.csv",
    index=False
)

print("Metadata created successfully!")
print("Total samples:", len(metadata))
print("Unique sample IDs:", metadata["sample_id"].nunique())
print("\nSubtype counts:")
print(metadata["subtype"].value_counts())
print("\nFirst 5 rows:")
print(metadata.head())
