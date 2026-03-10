# Description of Data

## Cell types
 12 cell types from two panels
### Immune Panel
 - Neutrophil
 - Macrophage
 - Cytotoxic T Cell
 - T Helper Cell
 - Treg Cell
 - Epithelium (imm)
### Stromal Panel
 - Periostin
 - CD146
 - CD34
 - SMA
 - Podoplanin
 - Epithelium (str)

### Number of Cell Groups (Excluding Epithelium)
 10 individual cell types
 45 pairs
 (10 choose 3 = 120) triples

### Number of Cell Groups (Including Epithelium)
 12 individual cell types
 66 pairs
 (10 choose 2 = 45) pairs not involving epithelium.
 5 pairs in each panel involving epithelium of that panel.
 Excluding any pair of cells where one is epi and the other comes from a different panel,
 we are left with 55 cell pairs.
 (10 choose 3 = 120) triples from non-epi cells.
 (5 choose 2 = 10) triples in the immune panel that all contain epithelium.
 (5 choose 2 = 10) triples in the stromal panel all containing epithelium.
 Total 140 triples.

## List of Features
### List of statistics
 For each bar in each barcode, compute:
 - avg
 - sd
 - med
 - range
 - p10
 - p25
 - p75
 - p90
 of the following quantities:
 - birth
 - death
 - midpt
 - length
 yielding 32 features, to which we also add
 - num_bars
 - entropy

### Diagrams
 For lone cell types we use (diagram, dimension)
 - (dom, 0)
 - (dom, 1)

 For pairs or triples we use
 - (ker, 0)
 - (ker, 1)
 - (im, 0)
 - (im, 1)
 - (cok, 1)

### Exclusions
 We also exclude the following features:
 - any statistic of birth, length, or midpt in domain and image of dimension 0

### Feature Count (Individual Cell Types)
 2 diagrams
 34 features per diagram
 (3*8=24) features excluded for dom0
 Total number of features = 2 * 34 - 24 = 44 features.

### Feature Count (Pairs or Triples)
 5 diagrams
 34 features per diagram
 Total = 170 features

## Size of Feature Space
### No Triples, Excluding Epithelium
 10 * 44 + 170 * 45 = 8090
### No Triples, Including Epithelium
 12 * 44 + 170 * 58 = 10388
### Triples, Excluding Epithelium
 10 * 44 + 170 * (45 + 20) = 11490
### Triples, Including Epithelium
 12 * 44 + 170 * (58 + 20) = 13788

## Samples
12274 samplies minus files with no celltype with at least 3 cells = 10295 samples
