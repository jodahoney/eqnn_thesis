## Noisy Backend Notes

Supported mixed-state noise models through the canonical [`NoiseConfig`](/Users/joda/Desktop/stanford/eqnn_thesis/eqnn/noise/config.py) path are:
- `none`
- `depolarizing`
- `amplitude_damping`
- `phase_damping` (with `dephasing` accepted as an alias)
- `coherent_overrotation`

For coherent over-rotation, the currently supported axis is `zz`, with modes:
- `fixed`
- `stochastic`
- `random_angle`
- `pair_dependent`

Qubit-dependent single-qubit noise is controlled by:
- `noise_application_scope`
  - `active`: only the active qubits in the current layer
  - `all`: all qubits after each layer
  - `selected_qubits`: only the configured noisy subset after each layer
- `noisy_qubits`
- `single_qubit_error_profile`

`single_qubit_error_profile` uses override semantics, not scaling semantics:
- if a profile value exists for qubit `i`, that value replaces the scalar single-qubit strength for that site
- if no profile value exists for `i`, the scalar channel parameter is used as the fallback

For `pair_dependent` coherent over-rotation, `pair_dependent_overrotation_angles` is indexed by the active-pair order in the current convolution layer, not by a physical qubit-pair label map.

In noisy comparison sweeps:
- `noisy_qubit_index=None` means the run uses the configured default scope/behavior
- an integer `noisy_qubit_index=k` means the run is forced into `selected_qubits` mode with `noisy_qubits=(k,)`

For `selected_qubits`, selected indices target qubits while they are present in the current effective register. After pooling or coarse-graining, selected indices outside the current effective register are skipped for that layer instead of treated as errors.

The optional symmetry diagnostic recorded by noisy comparison is an empirical prediction-drift diagnostic under sampled global SU(2) transformations. It is useful as a smoke-level robustness check, not a theoretical certificate.
