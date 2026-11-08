# Breast Cancer Subtype Classification
## Data Analysis Summary

### 1. Dataset
- Dataset: GSE96058 (SCAN-B)
- Primary samples: 3273
- Gene features: 30865
- Metadata records: 3409

### 2. Subtype Distribution
subtype
LumA      1657
LumB       729
Basal      339
Her2       327
Normal     221

### 3. Data Quality
- Missing values: 0
- Duplicate sample IDs: 0
- Variable genes: 28605

### 4. Differential Expression Analysis
- Statistical test: Kruskal-Wallis
- Multiple testing correction: Benjamini-Hochberg FDR
- Genes tested: 28604
- Significant genes (adjusted p < 0.05): 24365

### 5. Top Differentially Expressed Genes
      gene  kruskal_statistic  adjusted_p_value
    ZRANB1          348.16016               0.0
ZRANB2-AS1          880.41113               0.0
    IGFALS         1176.47120               0.0
      ZW10          231.97168               0.0
      ZXDA          624.66700               0.0
     MMP11          239.25781               0.0
     MMP12          573.42084               0.0
     MMP16          627.71875               0.0
     MMP17          503.66504               0.0
     MMP19          310.87890               0.0
      MMP2          448.75780               0.0
     MMP20          494.21260               0.0
    MMP23B          876.00100               0.0
     MMP24          353.57422               0.0
 MMP24-AS1          284.69140               0.0
   ZSCAN32          608.48145               0.0
     ZWINT         1290.59960               0.0
      A1BG          449.85645               0.0
  A1BG-AS1          399.09375               0.0
     AADAT          527.91504               0.0

### 6. Biomarker Candidate Genes
FOXC1, FOXA1, ESR1, GATA3, MLPH, SPDEF, AR, ABCC11, SOX10, KRT5, KRT14, FOXM1, UBE2C, BIRC5, AURKA, PTTG1, MYBL2, RRM2, BCL2, XBP1

These are exploratory candidates and require further validation.

### 7. Machine Learning Split
- Training samples: 2618
- Testing samples: 655
- Split ratio: 80:20
- Stratification: By molecular subtype

### 8. Generated Visualizations
- results/eda/subtype_distribution.png
- results/eda/expression_distribution.png
- results/eda/pca_subtypes.png
- results/biomarker_candidates/biomarker_heatmap.png