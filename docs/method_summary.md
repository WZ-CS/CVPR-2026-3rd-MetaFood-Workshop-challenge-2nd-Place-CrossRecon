# CrossRecon Method Summary

CrossRecon decomposes sparse-view food reconstruction into two complementary paths.

Path A reconstructs from real observations. In the full system, selected before/after
keyframes are segmented with GroundingDINO and SAM2, passed to Pi3 for feed-forward
geometry, aligned with the plate/support plane, and converted into a height-field mesh.
This path is trusted for volume because it preserves consumed or broken food geometry.

Path B reconstructs from a representative view. In the full system, a GPT Image
multi-view prompt synthesizes additional views, and Hunyuan3D decodes them into a
watertight mesh. This path is trusted for closed topology and visual mesh delivery,
but it can over-complete after-consumption food because the generative prior favors
complete objects.

Both paths share a metric-scale calibration module based on common table objects,
especially the plate diameter. Final volume is produced by visibility-aware and
state-aware fusion. The after state boosts Path A because Path B is more likely to
hallucinate missing food back into the mesh.

The repository includes a deterministic lite backend. It preserves the same data flow
and file outputs with classical image processing and parametric mesh generation, so the
pipeline can be executed without unpublished model weights. It is not intended to match
the challenge leaderboard numbers.

