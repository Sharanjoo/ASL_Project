macro_f1 (%) on the test split, mean ± std over seeds (Frame-only: n=5, +Interpolation: n=5, +Temporal: n=5, +Temporal & robust training: n=5, Raw-frame CNN: n=5)

| Condition | Frame-only | +Interpolation | +Temporal | +Temporal & robust training | Raw-frame CNN |
| --- | --- | --- | --- | --- | --- |
| clean | 47.1 ± 1.9 | 44.5 ± 1.4 | 43.7 ± 1.4 | 46.8 ± 3.1 | 36.7 ± 1.2 |
| occlusion_mild | 44.4 ± 2.0 | 42.4 ± 1.3 | 41.4 ± 2.2 | 44.7 ± 1.6 | 34.3 ± 2.8 |
| occlusion_moderate | 38.1 ± 1.2 | 38.2 ± 1.5 | 37.6 ± 1.7 | 41.1 ± 2.5 | 32.7 ± 1.6 |
| occlusion_severe | 29.6 ± 1.2 | 30.2 ± 1.7 | 27.3 ± 1.1 | 34.3 ± 1.3 | 21.4 ± 3.2 |
| blur_mild | 46.1 ± 1.3 | 45.0 ± 1.7 | 41.8 ± 1.2 | 45.7 ± 1.9 | 36.5 ± 1.8 |
| blur_moderate | 42.5 ± 1.3 | 41.4 ± 1.7 | 39.8 ± 2.2 | 42.8 ± 3.3 | 25.6 ± 2.0 |
| blur_severe | 28.2 ± 2.3 | 27.5 ± 1.3 | 29.6 ± 0.8 | 37.3 ± 1.4 | 13.9 ± 4.2 |
| missing_mild | 37.2 ± 1.9 | 37.7 ± 0.5 | 33.7 ± 2.7 | 44.2 ± 3.2 | 33.6 ± 1.9 |
| missing_moderate | 29.4 ± 2.9 | 34.1 ± 1.3 | 26.0 ± 4.6 | 41.1 ± 1.7 | 29.6 ± 1.5 |
| missing_severe | 11.5 ± 3.3 | 22.4 ± 1.4 | 9.6 ± 3.5 | 27.7 ± 1.6 | 17.7 ± 1.5 |
| **avg of 9 degraded** | 34.1 ± 1.3 | 35.4 ± 1.0 | 31.9 ± 1.1 | 39.9 ± 1.7 | 27.3 ± 0.7 |
