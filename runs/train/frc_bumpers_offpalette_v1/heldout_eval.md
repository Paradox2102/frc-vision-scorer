### Bumper detection by colour group, pooled over 4 matches
conf ≥ 0.35, stride 3; colour ok = matched box has the group's expected colour (teal/royal/navy → blue, red → red).

#### teal (expected blue)

| arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|
| before | 227 | 24.2% | 0.55 | 0.45 | 90.9% |
| after | 227 | 96.0% | 0.84 | 0.80 | 100.0% |
| Δ after − before |  | +71.8 pp | +0.289 | +0.350 | +9.1 pp |

#### royal blue (expected blue)

| arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|
| before | 3118 | 93.1% | 0.87 | 0.83 | 99.6% |
| after | 3118 | 92.8% | 0.87 | 0.85 | 99.3% |
| Δ after − before |  | -0.3 pp | +0.000 | +0.019 | -0.2 pp |

#### navy (expected blue)

| arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|
| before | 476 | 75.6% | 0.77 | 0.63 | 99.7% |
| after | 476 | 83.2% | 0.81 | 0.73 | 99.7% |
| Δ after − before |  | +7.6 pp | +0.040 | +0.107 | +0.0 pp |

#### red (expected red)

| arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|
| before | 3554 | 91.3% | 0.90 | 0.84 | 100.0% |
| after | 3554 | 92.9% | 0.88 | 0.86 | 99.9% |
| Δ after − before |  | +1.7 pp | -0.012 | +0.026 | -0.1 pp |

#### unclassified (expected colour not scored)

| arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|
| before | 744 | 34.1% | 0.70 | 0.58 | – |
| after | 744 | 50.1% | 0.76 | 0.65 | – |
| Δ after − before |  | +16.0 pp | +0.056 | +0.070 | – |

### Per match

#### 2026cabl_qm5 (841 frames)

| group | arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|---:|
| teal | before | 227 | 24.2% | 0.55 | 0.45 | 90.9% |
| teal | after | 227 | 96.0% | 0.84 | 0.80 | 100.0% |
| teal | Δ |  | +71.8 pp | +0.289 | +0.350 | +9.1 pp |
| royal blue | before | 17 | 47.1% | 0.58 | 0.42 | 100.0% |
| royal blue | after | 17 | 35.3% | 0.80 | 0.76 | 100.0% |
| royal blue | Δ |  | -11.8 pp | +0.214 | +0.342 | +0.0 pp |
| navy | before | 89 | 62.9% | 0.68 | 0.57 | 100.0% |
| navy | after | 89 | 87.6% | 0.77 | 0.73 | 100.0% |
| navy | Δ |  | +24.7 pp | +0.091 | +0.161 | +0.0 pp |
| red | before | 0 | – | – | – | – |
| red | after | 0 | – | – | – | – |
| red | Δ |  | – | – | – | – |
| unclassified | before | 293 | 33.4% | 0.65 | 0.51 | – |
| unclassified | after | 293 | 66.9% | 0.72 | 0.61 | – |
| unclassified | Δ |  | +33.4 pp | +0.068 | +0.102 | – |

#### 2026cabl_qm12 (843 frames)

| group | arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|---:|
| teal | before | 0 | – | – | – | – |
| teal | after | 0 | – | – | – | – |
| teal | Δ |  | – | – | – | – |
| royal blue | before | 4 | 75.0% | 0.64 | 0.55 | 66.7% |
| royal blue | after | 4 | 100.0% | 0.78 | 0.52 | 50.0% |
| royal blue | Δ |  | +25.0 pp | +0.135 | -0.031 | -16.7 pp |
| navy | before | 2 | 0.0% | – | – | – |
| navy | after | 2 | 0.0% | – | – | – |
| navy | Δ |  | +0.0 pp | – | – | – |
| red | before | 388 | 87.6% | 0.86 | 0.78 | 100.0% |
| red | after | 388 | 96.6% | 0.86 | 0.84 | 100.0% |
| red | Δ |  | +9.0 pp | +0.008 | +0.053 | +0.0 pp |
| unclassified | before | 9 | 11.1% | 0.65 | 0.65 | – |
| unclassified | after | 9 | 22.2% | 0.47 | 0.39 | – |
| unclassified | Δ |  | +11.1 pp | -0.184 | -0.262 | – |

#### 2026cur_qm103 (854 frames)

| group | arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|---:|
| teal | before | 0 | – | – | – | – |
| teal | after | 0 | – | – | – | – |
| teal | Δ |  | – | – | – | – |
| royal blue | before | 1299 | 94.1% | 0.87 | 0.83 | 99.4% |
| royal blue | after | 1299 | 93.5% | 0.87 | 0.85 | 99.2% |
| royal blue | Δ |  | -0.5 pp | +0.002 | +0.020 | -0.3 pp |
| navy | before | 328 | 79.6% | 0.79 | 0.63 | 100.0% |
| navy | after | 328 | 83.2% | 0.82 | 0.72 | 100.0% |
| navy | Δ |  | +3.7 pp | +0.030 | +0.092 | +0.0 pp |
| red | before | 1904 | 95.5% | 0.91 | 0.87 | 99.9% |
| red | after | 1904 | 96.4% | 0.89 | 0.88 | 99.8% |
| red | Δ |  | +0.8 pp | -0.019 | +0.004 | -0.2 pp |
| unclassified | before | 137 | 24.8% | 0.81 | 0.53 | – |
| unclassified | after | 137 | 30.7% | 0.80 | 0.60 | – |
| unclassified | Δ |  | +5.8 pp | -0.012 | +0.067 | – |

#### 2026cur_qm112 (851 frames)

| group | arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|---:|
| teal | before | 0 | – | – | – | – |
| teal | after | 0 | – | – | – | – |
| teal | Δ |  | – | – | – | – |
| royal blue | before | 1798 | 92.9% | 0.87 | 0.83 | 99.7% |
| royal blue | after | 1798 | 92.8% | 0.87 | 0.85 | 99.6% |
| royal blue | Δ |  | -0.1 pp | -0.002 | +0.018 | -0.1 pp |
| navy | before | 57 | 75.4% | 0.79 | 0.66 | 97.7% |
| navy | after | 57 | 78.9% | 0.82 | 0.79 | 97.8% |
| navy | Δ |  | +3.5 pp | +0.036 | +0.125 | +0.1 pp |
| red | before | 1262 | 86.0% | 0.88 | 0.76 | 100.0% |
| red | after | 1262 | 86.6% | 0.88 | 0.84 | 100.0% |
| red | Δ |  | +0.6 pp | -0.005 | +0.081 | +0.0 pp |
| unclassified | before | 305 | 39.7% | 0.77 | 0.62 | – |
| unclassified | after | 305 | 43.6% | 0.85 | 0.73 | – |
| unclassified | Δ |  | +3.9 pp | +0.075 | +0.115 | – |

### Focus teams

Team = replay slot label at the nearest meta frame; every mask sample of the team (no colour-group filter); colour ok vs the mask's alliance.

| team | match | arm | n | det % | conf med | conf p25 | colour ok % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4322 | pooled | before | 409 | 85.8% | 0.86 | 0.77 | 97.7% |
| 4322 | pooled | after | 409 | 95.1% | 0.86 | 0.83 | 97.4% |
| 4322 | pooled | Δ |  | +9.3 pp | +0.009 | +0.061 | -0.3 pp |
| 4322 | 2026cabl_qm5 | before | 8 | 87.5% | 0.67 | 0.64 | 14.3% |
| 4322 | 2026cabl_qm5 | after | 8 | 100.0% | 0.87 | 0.86 | 0.0% |
| 4322 | 2026cabl_qm5 | Δ |  | +12.5 pp | +0.205 | +0.222 | -14.3 pp |
| 4322 | 2026cabl_qm12 | before | 401 | 85.8% | 0.86 | 0.78 | 99.4% |
| 4322 | 2026cabl_qm12 | after | 401 | 95.0% | 0.86 | 0.83 | 99.5% |
| 4322 | 2026cabl_qm12 | Δ |  | +9.2 pp | +0.008 | +0.055 | +0.1 pp |
| 6800 | pooled | before | 819 | 80.0% | 0.85 | 0.72 | 92.1% |
| 6800 | pooled | after | 819 | 83.2% | 0.86 | 0.80 | 92.5% |
| 6800 | pooled | Δ |  | +3.2 pp | +0.013 | +0.073 | +0.4 pp |
| 6800 | 2026cur_qm103 | before | 373 | 63.5% | 0.77 | 0.62 | 78.1% |
| 6800 | 2026cur_qm103 | after | 373 | 68.6% | 0.81 | 0.71 | 80.1% |
| 6800 | 2026cur_qm103 | Δ |  | +5.1 pp | +0.038 | +0.087 | +2.0 pp |
| 6800 | 2026cur_qm112 | before | 446 | 93.7% | 0.87 | 0.80 | 100.0% |
| 6800 | 2026cur_qm112 | after | 446 | 95.3% | 0.88 | 0.85 | 100.0% |
| 6800 | 2026cur_qm112 | Δ |  | +1.6 pp | +0.009 | +0.058 | +0.0 pp |
| 498 | pooled | before | 365 | 38.9% | 0.65 | 0.50 | 92.3% |
| 498 | pooled | after | 365 | 68.8% | 0.73 | 0.65 | 92.8% |
| 498 | pooled | Δ |  | +29.9 pp | +0.080 | +0.150 | +0.6 pp |
| 498 | 2026cabl_qm5 | before | 365 | 38.9% | 0.65 | 0.50 | 92.3% |
| 498 | 2026cabl_qm5 | after | 365 | 68.8% | 0.73 | 0.65 | 92.8% |
| 498 | 2026cabl_qm5 | Δ |  | +29.9 pp | +0.080 | +0.150 | +0.6 pp |

### False boxes per frame

False = matched to no robot mask, centre in no mask bbox, IoS < 0.5 with every cached whole-robot box. Extra on-robot = unmatched but on a robot (duplicate / second box).

| arm | 2026cabl_qm5 false/f | 2026cabl_qm12 false/f | 2026cur_qm103 false/f | 2026cur_qm112 false/f | pooled false/f | pooled false total | pooled extra on-robot/f |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 1.353 | 0.421 | 0.219 | 0.456 | 0.610 | 2068 | 1.922 |
| after | 0.793 | 0.486 | 0.316 | 0.367 | 0.490 | 1659 | 2.036 |
| Δ after − before | -0.560 | +0.065 | +0.097 | -0.089 | -0.121 | -409 | +0.114 |

### Sanity: arm `before` (production weights) vs cached x-ray detections

All mask samples (no colour filter). Agreement = both matched or both unmatched; conf diff / same colour over samples matched in both.

| match | n | arm det % | cache det % | agreement | mean |Δconf| | same colour |
|---|---:|---:|---:|---:|---:|---:|
| 2026cabl_qm5 | 626 | 34.7% | 34.7% | 100.0% | 0.000 | 100.0% |
| 2026cabl_qm12 | 403 | 85.4% | 85.4% | 100.0% | 0.000 | 100.0% |
| 2026cur_qm103 | 3668 | 90.9% | 90.9% | 100.0% | 0.000 | 100.0% |
| 2026cur_qm112 | 3422 | 85.3% | 85.3% | 100.0% | 0.000 | 100.0% |
| pooled | 8119 | 84.0% | 84.0% | 100.0% | 0.000 | 100.0% |
