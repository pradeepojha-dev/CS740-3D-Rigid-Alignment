# CS740 - Comparing SVD and Quaternion Methods for 3D Point-Set Alignment

**Pradeep Ojha | Roll No. 26M0817 | Individual project | IIT Bombay, Autumn 2026**

This is the **cleaned submission package** for the CS740 project based on Eggert, Lorusso and Fisher (1997). It implements two algorithms (SVD and unit quaternion) for aligning *known corresponding* 3D point sets. It contains no generated datasets or figures because the experiments regenerate them.

## Code organization and file roles

| File | Purpose |
|---|---|
| `alignment.py` | Main mathematical implementation: input checks, centering, SVD rigid alignment, unit-quaternion rigid alignment, rotation conversion, evaluation metrics and singular-value ratio. |
| `demo.py` | Small example with a known 30-degree rotation and translation to verify recovery. |
| `experiments.py` | Generate synthetic 3D / nearly planar / nearly collinear point sets, add Gaussian target noise, evaluate both methods and save CSV summaries plus 6 figures to `results/`. |
| `test_alignment.py` | Ten automated tests checking noiseless recovery, rotation validity, planar/degenerate cases, reflections and basic input handling. |
| `requirements.txt` | Required Python dependencies (NumPy and Matplotlib). |
| `CS740_Presentation_26M0817.pptx` | Presentation covering problem, methods, experiments, results and limitations. |
| `github_logs.txt` | **Required separately:** authentic Git/GitHub commit history, to be added before submission. There is no such file in this package until exported from the actual repository. |

## How to run (Windows PowerShell)

```powershell
py -m pip install -r requirements.txt
py demo.py
py -m unittest -v test_alignment.py
py experiments.py
```

A successful experiment run generates `results/` with `all_trials.csv`, `summary.csv`, `summary.json`, and six PNG charts. These are reproducible outputs, **not necessary source files**, so they are excluded from the cleaned ZIP. All 10 automated tests passed in the review environment; also rerun them on your machine before submission.

## Experimental setup and findings

- Three geometric point distributions: 3D, nearly planar and nearly collinear.
- 12 / 40 / 100 points; target Gaussian noise σ = 0, .01, .03, .06 and .10; 25 trials per configuration, with random seed `7402026`.
- Total = 3 geometries × 3 counts × 5 noise levels × 25 trials × 2 algorithms = **2,250 fits**.
- At N=40 and noise σ=.06: mean rotation error was **0.630° (3D)**, **0.806° (nearly planar)**, and **44.588° (nearly collinear)** for each algorithm (shown precision). The nearly collinear case still had fitting RMSE around 0.10.
- The second/largest singular value ratio of centered source points was evaluated as an *exploratory warning* about near-collinearity. No general threshold or novelty is claimed.

## Assumptions and limitations

Input point matches are given and correct; transformation is rigid, without scaling; Gaussian noise is added only to target points. The evaluation is on synthetic data, not real camera measurements. Runtime rankings are machine-dependent and do not establish a universal faster algorithm. The original paper compares four algorithms; this individual project implements the SVD and unit-quaternion subset.

## Source materials

- D. W. Eggert, A. Lorusso, R. B. Fisher, *Estimating 3-D rigid body transformations: a comparison of four major algorithms*, Machine Vision and Applications 9:272-290, 1997.
- A. Rajwade, *Orthogonal Procrustes Problem*, CS740, IIT Bombay.

## GitHub logs (MANDATORY)

For this **one-person team**, put **one genuine `github_logs.txt`** in this folder with the commit history from the GitHub repository used for this project. Do not manufacture or backdate entries. On your computer, run `git log --all --date=iso-strict --pretty=fuller --stat > github_logs.txt` **inside the real Git repository** and copy the resulting file into this folder. Also include the repository URL at the top of that log. If this project was never in GitHub, create a truthful repository now, push the actual code, and export its genuine history. Only after this step is the ZIP ready under the new submission instructions.
