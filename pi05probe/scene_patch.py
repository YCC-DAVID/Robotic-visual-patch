"""Inject a thin, visual-only textured geom into a LIBERO scene, and generate
candidate anchor positions on an analytically-defined attachable plane.

Everything here rests on facts verified on a real libero_goal env; see
pi05probe/FINDINGS.md Q7/Q10. The non-obvious ones, because getting any of them
wrong fails silently rather than raising:

  * The geom MUST be wrapped in its own <body>. Appending a bare <geom> to
    <worldbody> renumbers every pre-existing geom_id (table_collision 7->8, ...),
    which invalidates any segmentation id recorded from a clean env.
  * group="1" is mandatory. LIBERO renders only geomgroup[1]; the default
    group 0 is invisible in BOTH rgb and segmentation (measured: 0 pixels).
  * contype=0 / conaffinity=0 so the arm cannot collide with it -- otherwise the
    action change mixes visual perturbation with contact dynamics.
  * z must clear the table by ~1mm. At z=0.9005 (bottom face flush with the
    0.900 table top) the pixel count becomes draw-order dependent.
  * The three fixtures (cabinet/stove/wine_rack) are NOT in qpos -- they live in
    sim.model.body_pos and are resampled from the global numpy RNG on EVERY
    reset(). So seed() must be called immediately before every reset(), or the
    "same scene" silently differs by ~20% of pixels.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET

import numpy as np


@dataclasses.dataclass(frozen=True)
class Plane:
    """An attachable surface, defined analytically (no mesh sampling)."""

    name: str
    origin: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    normal: np.ndarray
    bounds_u: tuple[float, float]
    bounds_v: tuple[float, float]

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Plane":
        return cls(
            name=cfg["name"],
            origin=np.asarray(cfg["origin"], dtype=float),
            tangent_u=_unit(cfg["tangent_u"]),
            tangent_v=_unit(cfg["tangent_v"]),
            normal=_unit(cfg["normal"]),
            bounds_u=tuple(cfg["bounds_u"]),
            bounds_v=tuple(cfg["bounds_v"]),
        )

    def to_world(self, u: float, v: float, lift: float = 0.0) -> np.ndarray:
        return self.origin + u * self.tangent_u + v * self.tangent_v + lift * self.normal

    def corners_uv(self, u: float, v: float, w: float, h: float) -> np.ndarray:
        du, dv = w / 2.0, h / 2.0
        return np.array([[u - du, v - dv], [u + du, v - dv], [u + du, v + dv], [u - du, v + dv]])

    def contains_uv(self, u: float, v: float, w: float, h: float) -> bool:
        c = self.corners_uv(u, v, w, h)
        return bool(
            (c[:, 0] >= self.bounds_u[0]).all()
            and (c[:, 0] <= self.bounds_u[1]).all()
            and (c[:, 1] >= self.bounds_v[0]).all()
            and (c[:, 1] <= self.bounds_v[1]).all()
        )


@dataclasses.dataclass(frozen=True)
class Anchor:
    index: int
    plane: str
    u: float
    v: float
    world: np.ndarray
    inside_plane: bool
    keepout_hits: tuple[str, ...]

    @property
    def legal(self) -> bool:
        return self.inside_plane and not self.keepout_hits


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError(f"zero-length vector: {v}")
    return a / n


def _overlaps(lo_a, hi_a, lo_b, hi_b) -> bool:
    return (lo_a < hi_b) and (lo_b < hi_a)


def make_anchors(cfg: dict) -> list[Anchor]:
    """Grid in plane-local 2D, mapped back to world. Orientation = plane normal."""
    plane = Plane.from_cfg(cfg["plane"])
    g = cfg["grid"]
    w, h = cfg["patch"]["size_wh"]
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]

    us = np.linspace(g["range_u"][0], g["range_u"][1], int(g["n_u"]))
    vs = np.linspace(g["range_v"][0], g["range_v"][1], int(g["n_v"]))

    anchors: list[Anchor] = []
    idx = 0
    for v in vs:
        for u in us:
            hits = []
            for name, box in (cfg.get("keepout") or {}).items():
                if _overlaps(u - w / 2, u + w / 2, box["x"][0], box["x"][1]) and _overlaps(
                    v - h / 2, v + h / 2, box["y"][0], box["y"][1]
                ):
                    hits.append(name)
            anchors.append(
                Anchor(
                    index=idx,
                    plane=plane.name,
                    u=float(u),
                    v=float(v),
                    world=plane.to_world(u, v, lift),
                    inside_plane=plane.contains_uv(u, v, w, h),
                    keepout_hits=tuple(hits),
                )
            )
            idx += 1
    return anchors


def make_xml_processor(cfg: dict, world_pos, texture_path: str, quat="1 0 0 0"):
    """Return an xml processor that injects one visual-only textured box geom.

    Pass to `env.env.set_xml_processor(...)` (note: env.env -- ControlEnv does not
    forward this method). It re-runs on every hard reset, so it survives reset().
    """
    name = cfg["patch"]["name"]
    w, h = cfg["patch"]["size_wh"]
    half = (w / 2.0, h / 2.0, cfg["patch"]["thickness"] / 2.0)
    px, py, pz = (float(c) for c in world_pos)

    def processor(xml: str) -> str:
        root = ET.fromstring(xml)
        asset = root.find("asset")
        ET.SubElement(asset, "texture", {"type": "2d", "name": f"{name}_tex", "file": texture_path})
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"{name}_mat",
                "texture": f"{name}_tex",
                "texrepeat": "1 1",
                "texuniform": "false",
                "specular": "0.0",
                "shininess": "0.0",
                "reflectance": "0.0",
            },
        )
        # Wrapping in a fresh <body> keeps the new geom LAST, preserving every
        # pre-existing geom_id. A bare <geom> in <worldbody> would shift them all.
        body = ET.SubElement(
            root.find("worldbody"),
            "body",
            {"name": f"{name}_body", "pos": f"{px} {py} {pz}", "quat": quat},
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": name,
                "type": "box",  # thin box, NOT a zero-thickness plane
                "size": f"{half[0]} {half[1]} {half[2]}",
                "pos": "0 0 0",
                "group": "1",  # mandatory, else invisible in rgb AND segmentation
                "contype": "0",  # visual only -- no collision
                "conaffinity": "0",
                "density": "0",
                "material": f"{name}_mat",
            },
        )
        return ET.tostring(root, encoding="unicode")

    return processor


def patch_geom_id(env, cfg: dict) -> int:
    return int(env.sim.model.geom_name2id(cfg["patch"]["name"]))


def visible_px(obs: dict, cam: str, gid: int) -> int:
    """Count pixels of the patch geom. Element-level seg value == geom_id, no offset."""
    return int((obs[f"{cam}_segmentation_element"][..., 0] == gid).sum())


def body_geom_ids(env, body_prefix: str) -> list[int]:
    """All geom ids whose name starts with body_prefix (for occlusion accounting)."""
    m = env.sim.model
    out = []
    for gid in range(m.ngeom):
        nm = m.geom_id2name(gid)
        if nm is not None and nm.startswith(body_prefix):
            out.append(gid)
    return out
