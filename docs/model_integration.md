# Full Model Integration Points

The runnable default backend is `lite`. Full challenge reproduction should replace the
following modules with the official model calls once weights and API access are provided.

## Segmentation

File: `src/crossrecon/backends/full.py`

Class: `GroundingDINOSAM2Backend`

Expected replacement:

- Run GroundingDINO with prompts containing the food name plus `plate`, `hand`,
  `fork`, `spoon`, `knife`, and `chopsticks`.
- Initialize SAM2 video propagation from the selected detections.
- Return food and support masks in the same `SegmentationResult` format used by the
  lite backend.

## Feed-forward Geometry

File: `src/crossrecon/backends/full.py`

Class: `Pi3GeometryBackend`

Expected replacement:

- Accept the selected keyframes for one state.
- Run Pi3 to obtain metric-free pointmaps and camera poses.
- Project food and plate masks into the point cloud.
- Fit the plate plane, build the support coordinate system, and return the food point
  cloud used by Path A height-field integration.

## Generative Reconstruction

File: `src/crossrecon/backends/full.py`

Class: `GPTImageHunyuanBackend`

Expected replacement:

- Crop the representative food view with its SAM2 mask.
- Generate six consistent views with GPT Image 2 or the final approved image model.
- Run Hunyuan3D to decode a watertight mesh.
- Align the mesh to Path A with OBB initialization and one-way Chamfer refinement.

## Configuration

Start from `configs/full_model_template.yaml` and fill the weight/config paths. The
current repository intentionally keeps those fields empty because the weights and API
credentials are expected to be provided later.

