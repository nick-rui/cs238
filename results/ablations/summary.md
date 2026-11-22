| Family | Variant | Loss | Accuracy (%) | Path log loss | Failures | Δ loss | Δ accuracy (pp) | Δ path log loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SGD | fixed | 0.3743 | 88.75 | -0.3383 | 0/200 | +0.0293 | -0.68 | +0.0331 |
| SGD | gradient | 0.3450 | 89.44 | -0.3714 | 0/400 | +0.0000 | +0.00 | +0.0000 |
| SGD | lr_only | 0.3672 | 88.85 | -0.3568 | 1/400 | +0.0222 | -0.59 | +0.0146 |
| SGD | fixed_memory | 0.3689 | 88.68 | -0.3433 | 3/400 | +0.0240 | -0.76 | +0.0281 |
| SGD | no_feedback | 0.3432 | 89.50 | -0.3661 | 0/400 | -0.0018 | +0.06 | +0.0053 |
| Adam | fixed | 0.2720 | 91.62 | -0.6971 | 0/200 | -0.0308 | +0.75 | -0.1091 |
| Adam | gradient | 0.3027 | 90.87 | -0.5880 | 0/400 | +0.0000 | +0.00 | +0.0000 |
| Adam | lr_only | 0.2375 | 92.76 | -0.6779 | 0/400 | -0.0653 | +1.89 | -0.0899 |
| Adam | fixed_memory | 0.2401 | 92.68 | -0.7041 | 0/400 | -0.0626 | +1.81 | -0.1162 |
| Adam | no_feedback | 0.3454 | 89.70 | -0.4950 | 0/400 | +0.0426 | -1.17 | +0.0930 |

Descriptive means over all ten original orderings and twenty episode seeds; two policy seeds. Differences are variant minus full, paired by episode seed, ordering, and policy seed; fixed uses the mean of both full policies within each episode/ordering. No significance claim.
