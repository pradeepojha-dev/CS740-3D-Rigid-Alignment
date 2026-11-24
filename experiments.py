"""Reproducible experiments for CS740.

Run: python experiments.py
Creates results/ with CSV files and PNG plots.
"""
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alignment import (estimate_svd, estimate_quaternion, transform,
                       axis_angle_to_rotation, rotation_error_degrees,
                       alignment_rmse, geometry_ratio)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 7402026
TRIALS = 25
NOISE_LEVELS = (0, 0.01, 0.03, 0.06, 0.10)
POINT_COUNTS = (12, 40, 100)
METHODS = {"SVD": estimate_svd, "Unit quaternion": estimate_quaternion}
TRUE_R = axis_angle_to_rotation([0.3, -0.5, 0.8], 42.0)
TRUE_T = np.array([0.4, -0.6, 0.25])


def generate_source(rng, count, geometry):
    p = rng.normal(size=(count, 3))
    if geometry == "3D":
        pass
    elif geometry == "nearly planar":
        p[:, 2] *= 0.02
    elif geometry == "nearly collinear":
        p[:, 1:] *= 0.01
    else:
        raise ValueError(geometry)
    return p


def save_rows(filename, rows, fieldnames):
    with (OUT / filename).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    rng = np.random.default_rng(SEED)
    records = []
    for geometry in ("3D", "nearly planar", "nearly collinear"):
        for count in POINT_COUNTS:
            for noise in NOISE_LEVELS:
                for trial in range(TRIALS):
                    x = generate_source(rng, count, geometry)
                    y_exact = transform(x, TRUE_R, TRUE_T)
                    # Source positions are exact, target positions contain independent Gaussian noise.
                    y = y_exact + rng.normal(loc=0., scale=noise, size=y_exact.shape)
                    ratio = geometry_ratio(x)
                    for label, solve in METHODS.items():
                        begin_ns = perf_counter_ns()
                        r, t = solve(x, y)
                        elapsed_us = (perf_counter_ns() - begin_ns) / 1_000
                        records.append(dict(
                            geometry=geometry, points=count, noise=noise, trial=trial,
                            method=label, singular_ratio=ratio,
                            rotation_error_deg=rotation_error_degrees(r, TRUE_R),
                            translation_error=float(np.linalg.norm(t - TRUE_T)),
                            point_rmse=alignment_rmse(x, y, r, t),
                            # Distance between aligned observed source and ideal noise-free target.
                            clean_target_rmse=alignment_rmse(x, y_exact, r, t),
                            orthogonality_error=float(np.linalg.norm(r.T @ r - np.eye(3), ord='fro')),
                            determinant=float(np.linalg.det(r)),
                            runtime_us=elapsed_us,
                        ))
    fieldnames = list(records[0])
    save_rows("all_trials.csv", records, fieldnames)

    summary = []
    for geo in ("3D", "nearly planar", "nearly collinear"):
        for count in POINT_COUNTS:
            for noise in NOISE_LEVELS:
                for method in METHODS:
                    subset = [r for r in records if r['geometry']==geo and r['points']==count and r['noise']==noise and r['method']==method]
                    item = dict(geometry=geo, points=count, noise=noise, method=method, trials=len(subset))
                    for key in ("rotation_error_deg", "translation_error", "point_rmse", "clean_target_rmse", "singular_ratio"):
                        item["mean_" + key] = float(np.mean([r[key] for r in subset]))
                    item["median_runtime_us"] = float(np.median([r["runtime_us"] for r in subset]))
                    summary.append(item)
    save_rows("summary.csv", summary, list(summary[0]))

    # Graph 1: rotation errors; one line for each point geometry, fixed N=40.
    plt.figure(figsize=(8.5, 5.4))
    for geo in ("3D", "nearly planar", "nearly collinear"):
        for method in METHODS:
            group = [s for s in summary if s['geometry']==geo and s['points']==40 and s['method']==method]
            plt.plot([s['noise'] for s in group], [s['mean_rotation_error_deg'] for s in group],
                     marker='o' if method=='SVD' else 's', linestyle='-' if method=='SVD' else '--',
                     label=f'{geo} / {method}')
    plt.xlabel('Gaussian noise std. dev. (coordinate units)')
    plt.ylabel('Mean rotation error (degrees)')
    plt.title('Rotation error under different point geometries (N = 40)')
    plt.grid(True, alpha=.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "rotation_error_vs_noise.png", dpi=175)
    plt.close()

    # Graph 2: translation error for normal 3D data.
    plt.figure(figsize=(7.7, 4.7))
    for method in METHODS:
        group = [s for s in summary if s['geometry']=='3D' and s['points']==40 and s['method']==method]
        plt.plot([s['noise'] for s in group], [s['mean_translation_error'] for s in group], marker='o', label=method)
    plt.xlabel('Gaussian noise std. dev. (coordinate units)')
    plt.ylabel('Mean translation error (coordinate units)')
    plt.title('Translation error vs noise (well-spread 3D, N = 40)')
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "translation_error_vs_noise.png", dpi=175)
    plt.close()

    # Graph 3: the fitted residual under noisy data.
    plt.figure(figsize=(7.7, 4.7))
    for method in METHODS:
        group = [s for s in summary if s['geometry']=='3D' and s['points']==40 and s['method']==method]
        plt.plot([s['noise'] for s in group], [s['mean_point_rmse'] for s in group], marker='o', label=method)
    plt.xlabel('Gaussian noise std. dev. (coordinate units)')
    plt.ylabel('Mean alignment RMSE (coordinate units)')
    plt.title('Residual point fitting error vs noise (3D, N = 40)')
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "rmse_vs_noise.png", dpi=175)
    plt.close()

    # Graph 4: reliability experiment on a fixed noise level / N.
    plt.figure(figsize=(7.6, 5.0))
    for geo in ("3D", "nearly planar", "nearly collinear"):
        points = [r for r in records if r['geometry']==geo and r['points']==40 and r['noise']==0.06 and r['method']=='SVD']
        plt.scatter([r['singular_ratio'] for r in points], [r['rotation_error_deg'] for r in points],
                    alpha=0.75, label=geo)
    plt.xlabel('sigma_2 / sigma_1 of centered source points')
    plt.ylabel('Rotation error (degrees)')
    plt.title('Does a small singular-value ratio indicate unreliable rotation?')
    plt.grid(True, alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "singular_ratio_reliability.png", dpi=175)
    plt.close()

    # Graph 5: timing. Microsecond-scale Python timings are approximate and environment-dependent.
    plt.figure(figsize=(7.7, 4.7))
    for method in METHODS:
        group = [s for s in summary if s['geometry']=='3D' and s['noise']==0.03 and s['method']==method]
        plt.plot([s['points'] for s in group], [s['median_runtime_us'] for s in group], marker='o', label=method)
    plt.xlabel('Number of corresponded points')
    plt.ylabel('Median runtime per fit (microseconds)')
    plt.title('Runtime comparison (approximate, machine-dependent)')
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "runtime_vs_points.png", dpi=175)
    plt.close()

    # Graph 6: 3D point alignment demo, before/after.
    vizrng = np.random.default_rng(SEED+1)
    x = generate_source(vizrng, 15, "3D")
    y = transform(x, TRUE_R, TRUE_T) + vizrng.normal(scale=.03, size=x.shape)
    r, t = estimate_svd(x, y)
    aligned = transform(x, r, t)
    fig = plt.figure(figsize=(7.8, 5.8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], label='Source points', marker='o')
    ax.scatter(y[:,0], y[:,1], y[:,2], label='Noisy target points', marker='^')
    ax.scatter(aligned[:,0], aligned[:,1], aligned[:,2], label='SVD aligned source', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Example 3D point-set alignment (15 points, noise = 0.03)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / 'point_alignment_example.png', dpi=175)
    plt.close()

    chosen = [s for s in summary if s['points']==40 and s['noise']==0.06]
    diagnostics = {
        'seed': SEED, 'trials_per_configuration': TRIALS,
        'runs': len(records), 'geometries': ['3D', 'nearly planar', 'nearly collinear'],
        'noise_std_levels': NOISE_LEVELS, 'point_counts': POINT_COUNTS,
        'configuration_table': chosen,
        'max_orthogonality_error': max(r['orthogonality_error'] for r in records),
        'max_abs_determinant_error': max(abs(r['determinant']-1) for r in records),
        'methods': list(METHODS),
    }
    with (OUT / 'summary.json').open('w') as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Completed {len(records)} fits = 3 geometries x 3 point counts x 5 noise levels x 25 trials x 2 algorithms")
    print('--- For N=40 and Gaussian sigma=0.06 (means of 25 trials) ---')
    print(f"{'Geometry':<17} {'Algorithm':<18} {'Rot(deg)':>10} {'Trans':>10} {'RMSE':>10} {'s2/s1':>10}")
    for s in chosen:
        print(f"{s['geometry']:<17} {s['method']:<18} {s['mean_rotation_error_deg']:>10.3f} "
              f"{s['mean_translation_error']:>10.4f} {s['mean_point_rmse']:>10.4f} {s['mean_singular_ratio']:>10.4f}")
    print(f"Max orthogonality error: {diagnostics['max_orthogonality_error']:.3e}")
    print(f"Max determinant deviation: {diagnostics['max_abs_determinant_error']:.3e}")
    print(f'Files saved in: {OUT}')

if __name__ == '__main__':
    main()
