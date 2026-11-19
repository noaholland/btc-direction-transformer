# Results

## Main run: 8h horizon

_Generated 2026-09-08 from `reports/`. Horizon 8h, walk-forward OOS 2025-01-01 to 2026-09-07, 7 quarterly folds._

### Walk-forward skill (every prediction made by a model that never saw that quarter)

| model | bars | accuracy | AUC | log-loss (coin flip = 0.6931) | Brier (coin flip = 0.25) | coverage at conf >= 0.65 | accuracy on covered |
|---|---|---|---|---|---|---|---|
| transformer ensemble | 29491 | 0.512 | 0.511 | 0.6933 | 0.2501 | 0.0% | n/a |
| logistic baseline (last bar) | 29491 | 0.514 | 0.513 | 0.6946 | 0.2507 | 0.9% | 0.641 |

Per fold (transformer acc / AUC, then logistic acc / AUC):

| fold | test quarter | train rows | TF acc | TF AUC | LR acc | LR AUC | temperature |
|---|---|---|---|---|---|---|---|
| 0 | 2025-01-01 to 2025-04-01 | 14496 | 0.501 | 0.507 | 0.512 | 0.521 | 1.00 |
| 1 | 2025-04-01 to 2025-07-01 | 18168 | 0.526 | 0.548 | 0.533 | 0.543 | 1.75 |
| 2 | 2025-07-01 to 2025-10-01 | 21880 | 0.531 | 0.518 | 0.510 | 0.504 | 1.00 |
| 3 | 2025-10-01 to 2026-01-01 | 25634 | 0.494 | 0.526 | 0.533 | 0.542 | 1.33 |
| 4 | 2026-01-01 to 2026-04-01 | 29388 | 0.477 | 0.505 | 0.504 | 0.498 | 1.70 |
| 5 | 2026-04-01 to 2026-07-01 | 33060 | 0.534 | 0.531 | 0.508 | 0.519 | 12.20 |
| 6 | 2026-07-01 to 2026-09-07 | 36772 | 0.524 | 0.498 | 0.490 | 0.476 | 2.99 |

### Calibration: does the confidence mean anything?

| confidence bin | bars | share of bars | realised accuracy | mean signed forward return (bps) |
|---|---|---|---|---|
| [0.50,0.55) | 27611 | 93.6% | 0.511 | +0.5 |
| [0.55,0.60) | 1769 | 6.0% | 0.520 | +15.5 |
| [0.60,0.65) | 111 | 0.4% | 0.586 | +18.4 |

### Skill at the bot's decision times only

| decision time (ET) | bars | AUC | median conf | 90th pct conf | max conf | n at >=0.55 | acc at >=0.55 | n at >=0.60 | acc at >=0.60 | n at >=0.65 |
|---|---|---|---|---|---|---|---|---|---|---|
| 09:00 | 614 | 0.478 | 0.512 | 0.544 | 0.605 | 37 | 0.459 | 1 | 0.000 | 0 |
| 10:00 | 614 | 0.485 | 0.512 | 0.544 | 0.604 | 35 | 0.514 | 2 | 0.500 | 0 |
| ALL | 29491 | 0.511 | 0.512 | 0.544 | 0.618 | 1880 | 0.523 | 111 | 0.586 | 0 |

### Backtest with the configured rules (Kelly risk mode, 40x, 0.008 stop / 0.016 take-profit, $1000 start)

* trades: **0**, final equity **$1000.00** (+0.0%)
* literal margin mode: 0 trades, final $1000.00
* buy & hold over the same window: -15.7%
* random-direction baseline (same timing/confidence, 200 seeds): median final $1000.00; 100% of random runs did at least as well as the model

### Threshold sensitivity (what happens if you lower the confidence bar)

| threshold | trades | win rate | final equity | max drawdown | daily Sharpe |
|---|---|---|---|---|---|
| 0.50 | 498 | 0.359 | $3.57 | -0.996 | -5.06 |
| 0.55 | 52 | 0.346 | $579.17 | -0.512 | -1.32 |
| 0.60 | 2 | 0.500 | $983.33 | -0.025 | -0.48 |
| 0.65 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.70 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.75 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.80 | 0 | n/a | $1000.00 | n/a | n/a |

Break-even win rate for this stop/take-profit structure is 1/(1+b) with b = 1.64, i.e. **37.9%**. Any threshold whose win rate sits below that loses money regardless of leverage; leverage only sets how fast.


## Sensitivity run: 24h horizon

_Generated 2026-09-08 from `h24/`. Horizon 24h, walk-forward OOS 2025-01-01 to 2026-09-06, 7 quarterly folds._

### Walk-forward skill (every prediction made by a model that never saw that quarter)

| model | bars | accuracy | AUC | log-loss (coin flip = 0.6931) | Brier (coin flip = 0.25) | coverage at conf >= 0.65 | accuracy on covered |
|---|---|---|---|---|---|---|---|
| transformer ensemble | 29459 | 0.523 | 0.498 | 0.6948 | 0.2508 | 0.0% | n/a |
| logistic baseline (last bar) | 29459 | 0.528 | 0.517 | 0.6951 | 0.2509 | 0.4% | 0.531 |

Per fold (transformer acc / AUC, then logistic acc / AUC):

| fold | test quarter | train rows | TF acc | TF AUC | LR acc | LR AUC | temperature |
|---|---|---|---|---|---|---|---|
| 0 | 2025-01-01 to 2025-04-01 | 14436 | 0.473 | 0.480 | 0.497 | 0.485 | 1.00 |
| 1 | 2025-04-01 to 2025-07-01 | 18108 | 0.540 | 0.557 | 0.590 | 0.596 | 12.20 |
| 2 | 2025-07-01 to 2025-10-01 | 21821 | 0.530 | 0.521 | 0.478 | 0.474 | 1.30 |
| 3 | 2025-10-01 to 2026-01-01 | 25575 | 0.531 | 0.552 | 0.551 | 0.561 | 1.00 |
| 4 | 2026-01-01 to 2026-04-01 | 29328 | 0.459 | 0.498 | 0.509 | 0.526 | 1.24 |
| 5 | 2026-04-01 to 2026-07-01 | 33000 | 0.575 | 0.561 | 0.546 | 0.562 | 12.20 |
| 6 | 2026-07-01 to 2026-09-07 | 36713 | 0.566 | 0.539 | 0.521 | 0.520 | 1.92 |

### Calibration: does the confidence mean anything?

| confidence bin | bars | share of bars | realised accuracy | mean signed forward return (bps) |
|---|---|---|---|---|
| [0.50,0.55) | 26478 | 89.9% | 0.524 | +7.7 |
| [0.55,0.60) | 2954 | 10.0% | 0.517 | +6.3 |
| [0.60,0.65) | 27 | 0.1% | 0.185 | -140.0 |

### Skill at the bot's decision times only

| decision time (ET) | bars | AUC | median conf | 90th pct conf | max conf | n at >=0.55 | acc at >=0.55 | n at >=0.60 | acc at >=0.60 | n at >=0.65 |
|---|---|---|---|---|---|---|---|---|---|---|
| 09:00 | 614 | 0.518 | 0.509 | 0.551 | 0.602 | 65 | 0.523 | 1 | 0.000 | 0 |
| 10:00 | 614 | 0.506 | 0.509 | 0.550 | 0.602 | 63 | 0.460 | 1 | 0.000 | 0 |
| ALL | 29459 | 0.498 | 0.509 | 0.550 | 0.603 | 2981 | 0.514 | 27 | 0.185 | 0 |

### Backtest with the configured rules (Kelly risk mode, 40x, 0.008 stop / 0.016 take-profit, $1000 start)

* trades: **0**, final equity **$1000.00** (+0.0%)
* literal margin mode: 0 trades, final $1000.00
* buy & hold over the same window: -15.7%
* random-direction baseline (same timing/confidence, 200 seeds): median final $1000.00; 100% of random runs did at least as well as the model

### Threshold sensitivity (what happens if you lower the confidence bar)

| threshold | trades | win rate | final equity | max drawdown | daily Sharpe |
|---|---|---|---|---|---|
| 0.50 | 564 | 0.358 | $3.60 | -0.997 | -4.33 |
| 0.55 | 95 | 0.368 | $316.28 | -0.761 | -1.80 |
| 0.60 | 2 | 0.000 | $867.36 | -0.133 | -0.77 |
| 0.65 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.70 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.75 | 0 | n/a | $1000.00 | n/a | n/a |
| 0.80 | 0 | n/a | $1000.00 | n/a | n/a |

Break-even win rate for this stop/take-profit structure is 1/(1+b) with b = 1.64, i.e. **37.9%**. Any threshold whose win rate sits below that loses money regardless of leverage; leverage only sets how fast.

