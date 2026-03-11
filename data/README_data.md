# Data Access Instructions

The raw genomic data used in this study are publicly available and can be retrieved
from the Bacterial and Viral Bioinformatics Resource Center (BV-BRC).

## Discovery Cohort

- **Source**: BV-BRC (https://www.bv-brc.org)
- **Organism**: *Klebsiella pneumoniae* (Taxon ID: 573)
- **BioProject**: PRJNA376414 (Houston Methodist Hospital clinical collection)
- **Initial isolates**: 1,522
- **After QC and deduplication**: 554 unique genomic profiles (413 resistant, 141 susceptible)

### Download steps

1. Go to https://www.bv-brc.org
2. Search for Taxon ID 573 (*Klebsiella pneumoniae*)
3. Filter by BioProject: PRJNA376414
4. Download AMR phenotype data and genome feature table
5. Export as CSV with `;` separator

### Preprocessing applied

- Excluded isolates with 'Intermediate' MIC phenotype (n=45)
- Quality control filter removing incomplete assemblies (n=180)
- Taxonomic re-verification (excluded *K. variicola* / *K. quasipneumoniae*)
- Feature space restricted to 195 curated AMR determinants
- Phenotype conflict filtering: removed profiles with discordant resistance labels
- Genomic deduplication: one representative per unique binary profile

## External Validation Cohort

Seven independent BioProjects were used for external validation (n=305 isolates total,
199 unique profiles after deduplication). See manuscript Table 1 for full accession list.

## Expected File Format

Both input CSV files use `;` as separator:
## Expected File Format

Both input CSV files use `;` as separator:

    Genome ID;[gene_1];[gene_2];...;[gene_195];Resistant Phenotype

- **Discovery file**: `asilverisetigenler.csv`
- **Validation file**: `validationPRJlison.csv`

The `Resistant Phenotype` column must contain exactly: `Resistant` or `Susceptible`
