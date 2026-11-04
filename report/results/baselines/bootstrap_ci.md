# Paired bootstrap over the 258 test clips

10000 resamples of clips, each clip scored as the share of 5 seeds that classify it correctly. 95% intervals in percentage points. `p(diff <= 0)` is the share of resamples where A is not better than B.

## Accuracy with 95% interval (%)

| Target | Frame-only (Stage 1) | Proposed (Stage 1 + refiner) | RTMW | MediaPipe |
| --- | --- | --- | --- | --- |
| clean | 49.5 [44.0, 54.8] | 48.8 [43.6, 54.1] | 53.8 [48.4, 59.0] | 37.0 [31.7, 42.1] |
| average of 9 degraded | 36.6 [32.7, 40.4] | 42.2 [37.8, 46.8] | 43.9 [39.6, 48.2] | 23.6 [20.7, 26.6] |
| severe blur | 31.5 [26.8, 36.4] | 40.0 [35.0, 45.3] | 37.4 [32.3, 42.6] | 8.4 [6.0, 11.1] |
| severe occlusion | 31.7 [26.8, 36.8] | 36.4 [31.3, 41.6] | 37.9 [32.9, 42.9] | 19.7 [15.7, 23.9] |
| severe missing frames | 13.8 [10.9, 16.9] | 29.5 [24.7, 34.3] | 23.7 [19.8, 27.8] | 13.7 [10.8, 16.9] |

## Paired differences A − B with 95% interval (points)

| Target | A | B | A − B | 95% interval | p(diff <= 0) | Excludes 0? |
| --- | --- | --- | --- | --- | --- | --- |
| clean | Proposed (Stage 1 + refiner) | Frame-only (Stage 1) | -0.6 | [-2.8, +1.6] | 0.722 | no |
| clean | RTMW | Frame-only (Stage 1) | +4.3 | [+0.4, +8.4] | 0.018 | yes |
| clean | RTMW | Proposed (Stage 1 + refiner) | +5.0 | [+1.2, +8.7] | 0.005 | yes |
| clean | Frame-only (Stage 1) | MediaPipe | +12.5 | [+7.9, +17.2] | 0.000 | yes |
| clean | RTMW | MediaPipe | +16.8 | [+12.1, +21.6] | 0.000 | yes |
| average of 9 degraded | Proposed (Stage 1 + refiner) | Frame-only (Stage 1) | +5.6 | [+3.9, +7.4] | 0.000 | yes |
| average of 9 degraded | RTMW | Frame-only (Stage 1) | +7.3 | [+4.8, +9.8] | 0.000 | yes |
| average of 9 degraded | RTMW | Proposed (Stage 1 + refiner) | +1.7 | [-1.0, +4.4] | 0.108 | no |
| average of 9 degraded | Frame-only (Stage 1) | MediaPipe | +13.0 | [+10.4, +15.7] | 0.000 | yes |
| average of 9 degraded | RTMW | MediaPipe | +20.4 | [+17.2, +23.6] | 0.000 | yes |
| severe blur | Proposed (Stage 1 + refiner) | Frame-only (Stage 1) | +8.5 | [+5.2, +11.9] | 0.000 | yes |
| severe blur | RTMW | Frame-only (Stage 1) | +5.9 | [+1.5, +10.4] | 0.005 | yes |
| severe blur | RTMW | Proposed (Stage 1 + refiner) | -2.6 | [-6.9, +1.6] | 0.895 | no |
| severe blur | Frame-only (Stage 1) | MediaPipe | +23.0 | [+18.3, +28.0] | 0.000 | yes |
| severe blur | RTMW | MediaPipe | +28.9 | [+23.7, +34.1] | 0.000 | yes |
| severe occlusion | Proposed (Stage 1 + refiner) | Frame-only (Stage 1) | +4.7 | [+1.2, +8.3] | 0.005 | yes |
| severe occlusion | RTMW | Frame-only (Stage 1) | +6.2 | [+2.4, +10.2] | 0.001 | yes |
| severe occlusion | RTMW | Proposed (Stage 1 + refiner) | +1.5 | [-2.8, +5.7] | 0.259 | no |
| severe occlusion | Frame-only (Stage 1) | MediaPipe | +12.0 | [+7.5, +16.5] | 0.000 | yes |
| severe occlusion | RTMW | MediaPipe | +18.2 | [+13.3, +23.0] | 0.000 | yes |
| severe missing frames | Proposed (Stage 1 + refiner) | Frame-only (Stage 1) | +15.7 | [+11.6, +19.8] | 0.000 | yes |
| severe missing frames | RTMW | Frame-only (Stage 1) | +9.9 | [+6.8, +13.2] | 0.000 | yes |
| severe missing frames | RTMW | Proposed (Stage 1 + refiner) | -5.7 | [-9.8, -1.6] | 0.998 | yes |
| severe missing frames | Frame-only (Stage 1) | MediaPipe | +0.1 | [-3.0, +3.1] | 0.484 | no |
| severe missing frames | RTMW | MediaPipe | +10.0 | [+6.6, +13.3] | 0.000 | yes |
