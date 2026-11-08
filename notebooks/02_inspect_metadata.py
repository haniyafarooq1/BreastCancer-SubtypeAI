import gzip
import csv
import os

files = [
    "data/raw/GSE96058-GPL11154_series_matrix.txt.gz",
    "data/raw/GSE96058-GPL18573_series_matrix.txt.gz"
]

for file_path in files:
    print("\nFILE:", os.path.basename(file_path))

    accessions = []
    titles = []

    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                accessions = next(csv.reader([line], delimiter="\t"))[1:]

            elif line.startswith("!Sample_title"):
                titles = next(csv.reader([line], delimiter="\t"))[1:]

    print("First 10 sample mappings:")
    for accession, title in zip(accessions[:10], titles[:10]):
        print(accession, "=", title)