# Cross-biosample transfer, stratified by element class

Run `20261119_irg-04-transfer-by-class` (Kaggle v1, XGBoost on GPU, 577 s). Same protocol as notebook 03 (canonical split, k=4 GC-matched, 3 seeds).
Each class of test positives is scored against **all** test negatives of that biosample. Bootstrap CIs are in `transfer_by_class.csv` (not reproduced here); cells with n < 30 are omitted.

## Why this experiment
Notebook 03 found much higher AUROC on test positives that are also positive in the training biosample ("shared") than on biosample-specific ones. Shared elements are probably promoter-rich, so that gap could just reflect class composition.

## 1. Element class strongly drives difficulty (in-domain AUROC)
| test    |   ALL |   PLS |   pELS |   dELS |
|:--------|------:|------:|-------:|-------:|
| gm12878 | 0.823 | 0.914 |  0.846 |  0.73  |
| k562    | 0.85  | 0.931 |  0.874 |  0.732 |
| tcell   | 0.87  | 0.935 |  0.886 |  0.744 |

Distal enhancers (dELS) are the hard class (about 0.73), promoters (PLS) are easy (0.91-0.94). A pooled AUROC mostly reflects the class mix.

## 2. Shared elements are promoter-rich
|                |   PLS shared |   PLS specific |   dELS shared |   dELS specific |
|:---------------|-------------:|---------------:|--------------:|----------------:|
| tcell->k562    |        0.499 |          0.232 |         0.117 |           0.284 |
| tcell->gm12878 |        0.458 |          0.203 |         0.162 |           0.344 |
| k562->tcell    |        0.51  |          0.35  |         0.108 |           0.252 |
| k562->gm12878  |        0.366 |          0.079 |         0.109 |           0.549 |
| gm12878->tcell |        0.461 |          0.354 |         0.15  |           0.249 |
| gm12878->k562  |        0.36  |          0.132 |         0.109 |           0.448 |

(Fractions of test positives in each group; e.g. K562 -> GM12878: shared elements are 37 % PLS and 11 % dELS, specific ones 8 % PLS and 55 % dELS.)

## 3. Shared vs specific within each class
| train   | test    | class   |   shared |   specific |   gap |   n_shared |   n_specific |
|:--------|:--------|:--------|---------:|-----------:|------:|-----------:|-------------:|
| gm12878 | k562    | PLS     |    0.941 |      0.845 | 0.095 |       1479 |          506 |
| gm12878 | k562    | dELS    |    0.792 |      0.686 | 0.105 |        450 |         1713 |
| gm12878 | k562    | pELS    |    0.905 |      0.796 | 0.11  |       2183 |         1601 |
| gm12878 | tcell   | PLS     |    0.941 |      0.928 | 0.013 |        268 |         1992 |
| gm12878 | tcell   | dELS    |    0.808 |      0.737 | 0.07  |         87 |         1402 |
| gm12878 | tcell   | pELS    |    0.91  |      0.877 | 0.033 |        226 |         2236 |
| k562    | gm12878 | PLS     |    0.947 |      0.794 | 0.152 |       1503 |          332 |
| k562    | gm12878 | dELS    |    0.786 |      0.65  | 0.137 |        450 |         2300 |
| k562    | gm12878 | pELS    |    0.909 |      0.756 | 0.153 |       2159 |         1556 |
| k562    | tcell   | PLS     |    0.947 |      0.93  | 0.017 |        274 |         1986 |
| k562    | tcell   | dELS    |    0.82  |      0.691 | 0.129 |         58 |         1431 |
| k562    | tcell   | pELS    |    0.928 |      0.88  | 0.048 |        205 |         2257 |
| tcell   | gm12878 | PLS     |    0.94  |      0.912 | 0.028 |        266 |         1569 |
| tcell   | gm12878 | dELS    |    0.787 |      0.688 | 0.099 |         94 |         2656 |
| tcell   | gm12878 | pELS    |    0.912 |      0.83  | 0.082 |        221 |         3494 |
| tcell   | k562    | PLS     |    0.947 |      0.914 | 0.033 |        268 |         1717 |
| tcell   | k562    | dELS    |    0.851 |      0.678 | 0.173 |         63 |         2100 |
| tcell   | k562    | pELS    |    0.928 |      0.845 | 0.083 |        206 |         3578 |

## 4. How much of the gap is class composition?
|                      |   overall gap (all classes pooled) |   mean within-class gap (classes with n_shared>=200) |   share of gap remaining within class |
|:---------------------|-----------------------------------:|-----------------------------------------------------:|--------------------------------------:|
| ('gm12878', 'k562')  |                              0.152 |                                                0.103 |                                  0.68 |
| ('gm12878', 'tcell') |                              0.049 |                                                0.023 |                                  0.47 |
| ('k562', 'gm12878')  |                              0.209 |                                                0.147 |                                  0.7  |
| ('k562', 'tcell')    |                              0.076 |                                                0.032 |                                  0.42 |
| ('tcell', 'gm12878') |                              0.107 |                                                0.055 |                                  0.51 |
| ('tcell', 'k562')    |                              0.115 |                                                0.058 |                                  0.5  |

## Interpretation
- **Class composition explains part of the gap, not all of it.** Shared positives score higher than specific ones in every one of the 18 class-by-pair cells. About 70 % of the pooled gap remains within class for the K562 <-> GM12878 pairs, but only about 42-51 % for the pairs involving T cell, where class mix explains roughly half. The within-class gap is 0.02-0.15 depending on the pair.
- **The effect is largest for dELS.** Specific dELS reach only 0.65-0.74 AUROC, shared dELS 0.79-0.85. Cell-type-specific enhancers are where a sequence model trained on another biosample does worst.
- A plausible reading (not tested here): shared elements are broadly active with stronger, more generic sequence signal, while biosample-specific elements depend on cell-type-specific TF motifs the training biosample did not teach. "Memorisation" is therefore the wrong word; the safer statement is that pooled transfer AUROC depends on the fraction of shared elements and class mix, so it overstates transfer to cell-type-specific regulation.
- **Reporting rule going forward:** report transfer per class and for biosample-specific positives separately.

## Limits
- T cell pairs have few shared dELS (n = 58-94), so those cells are noisy.
- Negatives are identical across groups; GC-matched negatives still allow a dinucleotide shortcut (see the unified-baselines document).
- One model family, fixed hyper-parameters, 3 biosamples.
