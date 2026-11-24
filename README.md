# CS740 - Comparing SVD and Quaternion Methods for 3D Point-Set Alignment

**Pradeep Ojha | 26M0817 | M.Tech CSE, IIT Bombay | Autumn 2026**  
Individual project for **CS740: Mathematics for Visual Computing**.

## Overview

This project implements two methods from Eggert, Lorusso and Fisher (1997) for estimating the rotation and translation between two sets of **corresponding 3D points**:

- **SVD-based alignment**
- **Unit-quaternion-based alignment**

I compare the methods using synthetic point sets, different amounts of Gaussian noise, and different point arrangements. I also examine whether the ratio of the two largest singular values can help flag nearly collinear point sets, where rotation estimation can become unreliable.

## Files

| File | Description |
| --- | --- |
| `alignment.py` | SVD and unit-quaternion implementations, input checks, and error metrics. |
| `demo.py` | Simple test with a known 30-degree rotation and translation. |
| `experiments.py` | Generates point sets, adds noise, runs comparisons, and saves results and plots. |
| `test_alignment.py` | Ten automated correctness tests. |
| `requirements.txt` | Python dependencies: NumPy and Matplotlib. |
| `CS740_Presentation_26M0817.pptx` | Project presentation. |
| `github_logs.txt` | Exported Git commit history, included **in the Moodle submission ZIP** (not kept in this repository). |

## Run the project

Open PowerShell in the project folder and run:

```powershell
py -m pip install -r requirements.txt
py demo.py
py -m unittest -v test_alignment.py
py experiments.py
```

Running `experiments.py` creates a `results/` directory with `all_trials.csv`, `summary.csv`, `summary.json`, and six PNG plots. These are generated files, so they are not included in the cleaned source-code submission.

## Experiments

I used three types of synthetic point arrangements: well-spread 3D, nearly planar, and nearly collinear.

| Setting | Values |
| --- | --- |
| Number of points | 12, 40, 100 |
| Gaussian noise standard deviation | 0, 0.01, 0.03, 0.06, 0.10 |
| Trials per setting | 25 |
| Random seed | `7402026` |
| Compared methods | SVD, unit quaternion |

This gives **2,250 algorithm fits** in total. The comparison includes rotation error, translation error, alignment RMSE, and execution time.

### Selected results

For **40 points** with **noise standard deviation 0.06**, mean rotation errors were:

| Point arrangement | SVD | Unit quaternion |
| --- | ---: | ---: |
| Well-spread 3D | 0.630° | 0.630° |
| Nearly planar | 0.806° | 0.806° |
| Nearly collinear | 44.588° | 44.588° |

Both methods had almost identical accuracy in these tests. Rotation estimation was much less reliable for nearly collinear points, even when point-fitting RMSE remained fairly small. In the measured runs, SVD was slightly faster, but timings depend on the machine and implementation.

For the additional experiment, I calculated the ratio `sigma_2 / sigma_1` from the singular values of the centered source points. A small ratio was associated with nearly collinear point sets and less reliable rotation estimates in these tests. I do **not** claim that it is a universal threshold.

## Assumptions and limitations

- Corresponding point pairs are already known and correctly matched.
- Only rotation and translation are estimated; scaling is not included.
- Gaussian noise is added to the target points.
- Experiments use synthetic data, not real camera measurements.
- Outliers and incorrect correspondences are not tested.
- The reference paper compares four algorithms; this individual project implements two of them.

## References

1. D. W. Eggert, A. Lorusso, and R. B. Fisher, *Estimating 3-D rigid body transformations: a comparison of four major algorithms*, Machine Vision and Applications, 9, 272–290 (1997).
2. A. Rajwade, *Orthogonal Procrustes Problem*, CS740, IIT Bombay.

## Repository and submission

Repository: <https://github.com/pradeepojha-dev/CS740-3D-Rigid-Alignment>

The final Moodle ZIP contains the cleaned source files, this README, the presentation, and `github_logs.txt` exported from the actual Git history.
