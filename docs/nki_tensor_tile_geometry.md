# Experimental per-Dot Tensor geometry response

`CostModel(tensor_tile_geometry_calibration=calibration)` opts into a geometry
response before timeline scheduling. The default remains the existing Tensor
calibration. Construct the calibration with explicit coefficients or with
`TensorTileGeometryCalibration.from_source_geometry_csv(path)`.

Each normalized `(M,K) @ (K,N)` Dot receives its own tile/view work estimate.
The reference unit is `(128,128,512)`; ceil-divided dimensions determine
reference-equivalent work units. Explicit source storage/range/shape identities
amortize view terms, with input versions distinguishing changed contents.
Unresolved identities remain occurrence-specific. Startup is charged once per
dtype. The simulation copies inputs and overrides stale aggregate Dot durations,
so running one mode does not contaminate a later run in another mode.

Coefficients fitted to homogeneous reference tiles do not establish transport
to partial tiles or mixed-stage kernels. Units are empirical features, not
verified compiler instruction counts or physical service prices. View sharing
does not establish hardware stationary reuse. Explicit Tensor transpose events
retain their ordinary cost; coefficients containing hidden transpose work may
therefore double count it. Validate this hypothesis on independent controls and
freeze predictions before evaluating downstream decisions. The result exposes
an enabled flag, reference-equivalent units, and an always-false transport
validation flag. No hardware accuracy or optimization gain is claimed by the
API or its behavioral tests.
