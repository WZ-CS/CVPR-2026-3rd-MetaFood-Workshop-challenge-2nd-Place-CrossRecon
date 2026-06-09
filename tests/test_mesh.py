import numpy as np

from crossrecon.mesh import heightfield_mesh, mesh_volume_m3, scale_mesh_to_volume


def test_heightfield_volume_metadata():
    height = np.ones((10, 10), dtype=float) * 0.02
    mesh = heightfield_mesh(height, step_x=0.01, step_y=0.01)
    assert abs(mesh.metadata["volume_m3"] - 0.0002) < 1e-12


def test_scale_mesh_to_volume_records_target():
    height = np.ones((8, 8), dtype=float) * 0.01
    mesh = heightfield_mesh(height, step_x=0.01, step_y=0.01)
    scaled = scale_mesh_to_volume(mesh, 0.0005)
    assert abs(scaled.metadata["volume_m3"] - 0.0005) < 1e-12
    assert mesh_volume_m3(scaled) > 0

