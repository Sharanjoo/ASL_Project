top5 (%) on the test split, mean ± std over seeds (Frame-only: n=5, +Interpolation: n=5, +Temporal: n=5, +Temporal & robust training: n=5, Raw-frame CNN: n=5)

| Condition | Frame-only | +Interpolation | +Temporal | +Temporal & robust training | Raw-frame CNN |
| --- | --- | --- | --- | --- | --- |
| clean | 80.9 ± 1.6 | 77.0 ± 1.9 | 79.1 ± 1.4 | 80.3 ± 1.3 | 71.2 ± 0.8 |
| occlusion_mild | 78.8 ± 1.3 | 75.1 ± 1.9 | 77.6 ± 1.6 | 78.2 ± 1.8 | 69.5 ± 1.4 |
| occlusion_moderate | 74.3 ± 1.7 | 71.2 ± 1.9 | 74.7 ± 0.7 | 76.4 ± 1.6 | 67.8 ± 3.2 |
| occlusion_severe | 65.0 ± 2.1 | 62.1 ± 2.2 | 61.4 ± 2.3 | 69.8 ± 2.9 | 54.3 ± 2.5 |
| blur_mild | 80.4 ± 1.3 | 76.1 ± 1.7 | 77.8 ± 1.2 | 79.5 ± 2.2 | 71.2 ± 1.1 |
| blur_moderate | 77.8 ± 1.9 | 73.1 ± 1.7 | 75.0 ± 1.1 | 76.8 ± 2.1 | 64.7 ± 3.8 |
| blur_severe | 65.8 ± 1.7 | 62.1 ± 1.9 | 67.5 ± 2.1 | 71.3 ± 1.7 | 44.6 ± 4.8 |
| missing_mild | 71.6 ± 3.5 | 69.8 ± 2.3 | 68.7 ± 2.9 | 77.8 ± 2.7 | 70.2 ± 1.6 |
| missing_moderate | 58.9 ± 5.1 | 64.3 ± 2.6 | 55.4 ± 3.4 | 74.5 ± 2.2 | 65.9 ± 2.5 |
| missing_severe | 35.2 ± 5.7 | 50.4 ± 3.6 | 33.2 ± 3.4 | 58.3 ± 1.8 | 50.5 ± 2.2 |
| **avg of 9 degraded** | 67.5 ± 2.0 | 67.1 ± 1.8 | 65.7 ± 1.2 | 73.6 ± 1.7 | 62.1 ± 1.7 |
