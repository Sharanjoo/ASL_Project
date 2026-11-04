top1 (%) on the test split, mean ± std over seeds (Frame-only: n=5, +Interpolation: n=5, +Temporal: n=5, +Temporal & robust training: n=5, Raw-frame CNN: n=5)

| Condition | Frame-only | +Interpolation | +Temporal | +Temporal & robust training | Raw-frame CNN |
| --- | --- | --- | --- | --- | --- |
| clean | 49.5 ± 1.3 | 46.7 ± 1.5 | 46.4 ± 1.5 | 48.8 ± 2.6 | 39.0 ± 1.6 |
| occlusion_mild | 46.9 ± 2.0 | 44.6 ± 1.6 | 44.3 ± 2.0 | 47.1 ± 1.0 | 36.9 ± 2.9 |
| occlusion_moderate | 41.0 ± 1.2 | 40.3 ± 1.5 | 39.9 ± 1.7 | 44.3 ± 2.6 | 34.7 ± 2.7 |
| occlusion_severe | 31.7 ± 0.8 | 31.7 ± 2.2 | 29.8 ± 1.2 | 36.4 ± 1.3 | 23.4 ± 3.1 |
| blur_mild | 48.7 ± 1.0 | 46.7 ± 2.5 | 44.6 ± 1.1 | 48.1 ± 1.6 | 38.7 ± 2.3 |
| blur_moderate | 45.0 ± 1.1 | 43.1 ± 1.8 | 42.3 ± 1.8 | 45.5 ± 3.1 | 28.8 ± 1.5 |
| blur_severe | 31.5 ± 2.1 | 29.8 ± 1.4 | 32.6 ± 0.8 | 40.0 ± 1.2 | 16.9 ± 3.5 |
| missing_mild | 39.1 ± 1.9 | 39.8 ± 0.6 | 37.0 ± 2.6 | 46.4 ± 2.6 | 36.2 ± 2.4 |
| missing_moderate | 31.7 ± 3.3 | 36.4 ± 1.4 | 28.2 ± 4.5 | 42.8 ± 1.9 | 32.6 ± 1.1 |
| missing_severe | 13.8 ± 3.1 | 25.0 ± 2.0 | 12.4 ± 2.5 | 29.5 ± 2.2 | 20.5 ± 0.9 |
| **avg of 9 degraded** | 36.6 ± 1.3 | 37.5 ± 1.4 | 34.6 ± 1.0 | 42.2 ± 1.7 | 29.9 ± 1.2 |
