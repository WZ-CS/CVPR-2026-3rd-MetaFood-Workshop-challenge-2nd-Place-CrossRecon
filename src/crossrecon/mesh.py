from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .types import Mesh


def mesh_volume_m3(mesh: Mesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    tri = vertices[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    volume = abs(float(signed.sum()))
    if volume <= 1e-15 and "volume_m3" in mesh.metadata:
        return float(mesh.metadata["volume_m3"])
    return volume


def scale_mesh_to_volume(mesh: Mesh, target_volume_m3: float) -> Mesh:
    current = mesh_volume_m3(mesh)
    if current <= 1e-15:
        scaled = Mesh(mesh.vertices.copy(), mesh.faces.copy(), dict(mesh.metadata))
        scaled.metadata["volume_m3"] = float(target_volume_m3)
        return scaled
    factor = (float(target_volume_m3) / current) ** (1.0 / 3.0)
    center = mesh.vertices.mean(axis=0, keepdims=True)
    vertices = (mesh.vertices - center) * factor + center
    metadata = dict(mesh.metadata)
    metadata["volume_m3"] = float(target_volume_m3)
    metadata["scale_to_volume_factor"] = float(factor)
    return Mesh(vertices=vertices, faces=mesh.faces.copy(), metadata=metadata)


def heightfield_mesh(
    height: np.ndarray,
    step_x: float,
    step_y: float,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    metadata: Optional[dict] = None,
) -> Mesh:
    height = np.asarray(height, dtype=np.float64)
    rows, cols = height.shape
    x0, y0 = origin_xy

    xs = x0 + (np.arange(cols) - (cols - 1) / 2.0) * float(step_x)
    ys = y0 + (np.arange(rows) - (rows - 1) / 2.0) * float(step_y)
    xx, yy = np.meshgrid(xs, ys)
    top = np.stack([xx, yy, height], axis=-1).reshape(-1, 3)
    bottom = np.stack([xx, yy, np.zeros_like(height)], axis=-1).reshape(-1, 3)
    vertices = np.vstack([top, bottom])

    def tid(i: int, j: int) -> int:
        return i * cols + j

    def bid(i: int, j: int) -> int:
        return rows * cols + i * cols + j

    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, d = tid(i, j), tid(i, j + 1), tid(i + 1, j), tid(i + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
            ba, bb, bc, bd = bid(i, j), bid(i, j + 1), bid(i + 1, j), bid(i + 1, j + 1)
            faces.append([ba, bb, bc])
            faces.append([bb, bd, bc])

    for j in range(cols - 1):
        faces.append([tid(0, j), tid(0, j + 1), bid(0, j)])
        faces.append([tid(0, j + 1), bid(0, j + 1), bid(0, j)])
        faces.append([tid(rows - 1, j), bid(rows - 1, j), tid(rows - 1, j + 1)])
        faces.append([tid(rows - 1, j + 1), bid(rows - 1, j), bid(rows - 1, j + 1)])

    for i in range(rows - 1):
        faces.append([tid(i, 0), bid(i, 0), tid(i + 1, 0)])
        faces.append([tid(i + 1, 0), bid(i, 0), bid(i + 1, 0)])
        faces.append([tid(i, cols - 1), tid(i + 1, cols - 1), bid(i, cols - 1)])
        faces.append([tid(i + 1, cols - 1), bid(i + 1, cols - 1), bid(i, cols - 1)])

    volume = float(height.sum() * step_x * step_y)
    mesh_metadata = dict(metadata or {})
    mesh_metadata.setdefault("volume_m3", volume)
    mesh_metadata.setdefault("mesh_type", "heightfield")
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), metadata=mesh_metadata)


def ellipsoid_mesh(
    radius_x: float,
    radius_y: float,
    radius_z: float,
    rows: int = 32,
    cols: int = 64,
    metadata: Optional[dict] = None,
) -> Mesh:
    vertices = []
    for i in range(rows + 1):
        theta = np.pi * i / rows
        z = radius_z * np.cos(theta)
        ring_r = np.sin(theta)
        for j in range(cols):
            phi = 2.0 * np.pi * j / cols
            vertices.append([radius_x * ring_r * np.cos(phi), radius_y * ring_r * np.sin(phi), z])

    faces = []
    for i in range(rows):
        for j in range(cols):
            a = i * cols + j
            b = i * cols + (j + 1) % cols
            c = (i + 1) * cols + j
            d = (i + 1) * cols + (j + 1) % cols
            faces.append([a, c, b])
            faces.append([b, c, d])

    mesh_metadata = dict(metadata or {})
    mesh_metadata.setdefault("mesh_type", "ellipsoid")
    mesh_metadata.setdefault("volume_m3", 4.0 / 3.0 * np.pi * radius_x * radius_y * radius_z)
    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        metadata=mesh_metadata,
    )


def write_obj(mesh: Mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CrossRecon mesh\n")
        for key, value in sorted(mesh.metadata.items()):
            f.write(f"# {key}: {value}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in mesh.faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")

