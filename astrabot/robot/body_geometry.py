"""Build conservative body boxes from the source URDF meshes, offline only."""

import argparse
import hashlib
import itertools
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

BODY_LINKS = ("pelvis", "pelvis_contour_link", "waist_yaw_link", "waist_roll_link", "torso_link")


def mesh_vertices(path):
    """Read binary or ASCII STL vertices, rejecting incomplete/nonfinite geometry."""
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError("invalid_binary_stl")
    count = struct.unpack_from("<I", data, 80)[0]
    if count and len(data) == 84 + 50 * count:
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).astype(float)
    else:
        try:
            lines = [line.split() for line in data.decode("ascii").splitlines() if line.strip()]
            faces = sum(line[0] == "endfacet" for line in lines)
            vertices = np.array([[float(value) for value in line[1:]] for line in lines if line[0] == "vertex"])
            if (
                lines[0][0] != "solid"
                or lines[-1][0] != "endsolid"
                or not faces
                or vertices.shape != (3 * faces, 3)
                or sum(line[0] == "facet" for line in lines) != faces
            ):
                raise ValueError("invalid_ascii_stl")
        except (UnicodeDecodeError, IndexError) as exc:
            raise ValueError("invalid_stl") from exc
    if not np.isfinite(vertices).all():
        raise ValueError("nonfinite_mesh")
    return vertices, hashlib.sha256(data).hexdigest()


def build(urdf_path, links=BODY_LINKS, include_hulls=False):
    """Bound every source vertex; convex triangles remain inside these boxes."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    entries = []
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")}
    for frame in links:
        link = root.find(f"link[@name='{frame}']")
        if link is None:
            raise ValueError(f"missing_body_link:{frame}")
        nodes = link.findall("collision")
        kind = "collision"
        if not nodes:
            nodes, kind = link.findall("visual"), "visual_fallback"
        if not nodes:
            raise ValueError(f"missing_body_geometry:{frame}")
        for node in nodes:
            mesh, cylinder, box = (node.find(f"geometry/{name}") for name in ("mesh", "cylinder", "box"))
            if mesh is not None:
                vertices, digest = mesh_vertices(urdf_path.parent / mesh.attrib["filename"])
                source = mesh.attrib["filename"]
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("invalid_mesh_scale")
                vertices *= scale
            elif cylinder is not None:
                radius, length = float(cylinder.get("radius")), float(cylinder.get("length"))
                if min(radius, length) <= 0 or not np.isfinite([radius, length]).all():
                    raise ValueError("invalid_cylinder")
                vertices = np.array(
                    list(itertools.product((-radius, radius), (-radius, radius), (-length / 2, length / 2)))
                )
                source, digest = "urdf:cylinder", None
            elif box is not None:
                size = np.fromstring(box.get("size"), sep=" ")
                if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                    raise ValueError("invalid_primitive_box")
                vertices = np.array(list(itertools.product(*zip(-size / 2, size / 2))))
                source, digest = "urdf:box", None
            else:
                raise ValueError(f"unsupported_body_geometry:{frame}")
            origin = node.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                vertices = vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            lower, upper = vertices.min(0), vertices.max(0)
            entries.append(
                {
                    "frame": frame,
                    "parent": parents.get(frame),
                    "source_geometry": kind,
                    "source_mesh": source,
                    "sha256": digest,
                    "vertex_count": len(vertices),
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                    "corners": list(itertools.product(*zip(lower.tolist(), upper.tolist()))),
                }
            )
            if include_hulls:
                hull = ConvexHull(vertices)
                entries[-1]["hull_vertices"] = vertices[hull.vertices]
    return {
        "source_urdf": urdf_path.name,
        "source_urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "method": "all source mesh vertices bounded in their URDF link frame; visual fallback where no collision exists",
        "entries": entries,
    }


def write_assets(urdf_path, output, hull_path, links=BODY_LINKS):
    """Save compact source-hull vertices separately from provenance metadata."""
    data = build(urdf_path, links=links, include_hulls=True)
    arrays = {}
    for index, entry in enumerate(data["entries"]):
        key = f"body_{index}"
        arrays[key] = entry.pop("hull_vertices").astype(np.float32)
        entry["hull_key"] = key
    np.savez_compressed(hull_path, **arrays)
    data["hull_file"] = Path(hull_path).name
    data["hull_file_sha256"] = hashlib.sha256(Path(hull_path).read_bytes()).hexdigest()
    data["method"] += "; narrow phase uses the complete source vertex convex hull"
    Path(output).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hulls", type=Path)
    parser.add_argument("--links", nargs="+", default=BODY_LINKS)
    args = parser.parse_args()
    if args.hulls:
        write_assets(args.urdf, args.output, args.hulls, links=args.links)
    else:
        args.output.write_text(json.dumps(build(args.urdf, links=args.links), indent=2) + "\n")
