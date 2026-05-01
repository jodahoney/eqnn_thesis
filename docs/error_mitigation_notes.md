## Error Mitigation Notes

These utilities are error mitigation methods, not full quantum error correction. They do not introduce encoded logical qubits, syndrome measurements, or recovery maps.

### Symmetry-Agnostic Post-Processing

Zero-noise extrapolation (ZNE) fits a metric measured at nonzero noise strengths and extrapolates the fitted curve back to noise strength 0. The current post-hoc utility supports linear and quadratic fits, optional maximum-noise filtering, and explicit noise-strength selection.

For the n=7 depolarizing-collapse benchmark, compare low-noise fits over strengths such as `0.0 0.01 0.03 0.05` against all-noise fits through `0.1`. ZNE can be unreliable when the fit includes the nonlinear collapse region around roughly `0.05` to `0.08`, because a low-order polynomial fit may turn the collapse into a misleading zero-noise intercept.

### Symmetry-Agnostic Training-Time Mitigation

Noise-aware training samples a training noise strength from a configured list at each epoch and trains under that noisy backend configuration for the epoch. Final evaluation still uses the requested evaluation noise grid. This is a generic robustness baseline for the n=7 depolarizing collapse, where useful training strengths include `0.0 0.01 0.03 0.05` and evaluation strengths include `0.0 0.03 0.05 0.08 0.1`.

### Symmetry-Aware Post-Processing

Symmetry-twirled evaluation averages predictions over sampled globally SU(2)-rotated versions of each input state:

```text
f_twirled(rho) = average_k f(g_k rho g_k^dagger)
```

This is evaluation-only. It does not change training, model parameters, or the noisy channel.

The existing empirical symmetry-drift diagnostic measures prediction changes under sampled global SU(2) rotations. In the observed n=7 depolarizing failure mode, that diagnostic can remain near zero even as accuracy collapses, so simple drift detection alone may not explain or fix the collapse. Twirled evaluation is still useful as a low-risk symmetry-aware post-processing check, especially when comparing SU2-QCNN behavior against HEA-QCNN baselines under noise strengths around `0.03` to `0.08`.

### Symmetry-Aware Training-Time Mitigation

Symmetry-regularized training augments the task objective with a sampled SU(2) prediction-consistency penalty:

```text
L_total = L_task + beta * average |f(rho) - f(g rho g^dagger)|^2
```

The implementation uses the same sampled global SU(2) rotation logic as the diagnostics and evaluates the penalty on a configurable subset of training states. It is an experimental robustness method, not a guarantee that the n=7 depolarizing collapse will be fixed. The empirical symmetry-drift diagnostic staying near zero during collapse means the failure may not be explained by simple global-SU(2) prediction drift alone.

Example Slurm entry points:
- [`run_mitigation_n7_depolarizing_noise_aware.sbatch`](/Users/joda/Desktop/stanford/eqnn_thesis/scripts/slurm/run_mitigation_n7_depolarizing_noise_aware.sbatch)
- [`run_mitigation_n7_depolarizing_symmetry_regularized.sbatch`](/Users/joda/Desktop/stanford/eqnn_thesis/scripts/slurm/run_mitigation_n7_depolarizing_symmetry_regularized.sbatch)
