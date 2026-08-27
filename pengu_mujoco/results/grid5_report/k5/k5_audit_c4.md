# c4 full-passer K=5 audit (391,285 rows; working notes)

Bias: rows exist only where the r=0 seed passed (pass_rate floor 0.2 by construction).

mu=0.1: passers=144043, joined K5=144043 (100.0%)

mu=0.3: passers=144223, joined K5=144223 (100.0%)

mu=0.5: passers=57950, joined K5=57950 (100.0%)

mu=0.7: passers=45069, joined K5=45069 (100.0%)

## Inflation and ranking survival

| mu | med K5/K1 (top-100) | K1-champ rank in K5-mean | top-20 overlap K1 vs K5-mean | top20 pass>=0.8 | top20 pass=1.0 | top20 min>0.05 |
|---|---|---|---|---|---|---|
| 0.1 | 0.92 | 8 | 9/20 | 19/20 | 17/20 | 19/20 |
| 0.3 | 0.97 | 265 | 5/20 | 13/20 | 11/20 | 16/20 |
| 0.5 | 0.87 | 954 | 1/20 | 17/20 | 13/20 | 17/20 |
| 0.7 | 0.99 | 213 | 10/20 | 19/20 | 18/20 | 18/20 |

## Whole-passer-population gate rates (all joined rows)

| mu | pass=1.0 | >=0.8 | >=0.6 | min>0.05 | min>0 |
|---|---|---|---|---|---|
| 0.1 | 86.0% | 92.0% | 95.8% | 86.9% | 96.5% |
| 0.3 | 81.9% | 89.5% | 94.3% | 83.9% | 90.7% |
| 0.5 | 56.9% | 73.2% | 84.6% | 59.4% | 81.7% |
| 0.7 | 57.8% | 73.3% | 84.6% | 60.2% | 82.5% |
