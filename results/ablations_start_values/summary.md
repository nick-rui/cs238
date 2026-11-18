| Family | Variant | Loss | Accuracy (%) | Path log loss | Failures | Δ loss | Δ accuracy (pp) | Δ path log loss | Δ loss vs tuned-held | Δ accuracy vs tuned-held (pp) | Δ path log loss vs tuned-held |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | gradient | 0.3450 | 89.44 | -0.3714 | 0/400 | +0.0000 | +0.00 | +0.0000 | — | — | — |
| SGD | lr_only | 0.3471 | 89.52 | -0.3475 | 0/400 | +0.0021 | +0.08 | +0.0239 | -0.0200 | +0.66 | +0.0093 |
| SGD | fixed_memory | 0.3648 | 88.83 | -0.3336 | 0/400 | +0.0198 | -0.60 | +0.0378 | -0.0041 | +0.16 | +0.0097 |
| Adam | gradient | 0.3027 | 90.87 | -0.5880 | 0/400 | +0.0000 | +0.00 | +0.0000 | — | — | — |
| Adam | lr_only | 0.3487 | 89.82 | -0.4482 | 0/400 | +0.0459 | -1.05 | +0.1398 | +0.1112 | -2.94 | +0.2296 |
| Adam | fixed_memory | 0.3378 | 89.99 | -0.5113 | 0/400 | +0.0351 | -0.88 | +0.0766 | +0.0977 | -2.69 | +0.1928 |

Descriptive means over all ten original orderings and twenty episode seeds; two policy seeds. Differences are start-held minus full (gradient) or start-held minus the same tuned-held ablation, paired by episode seed, ordering, and policy seed. No significance claim.
