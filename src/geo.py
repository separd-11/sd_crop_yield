"""Raion polygons for choropleths, shared by the figures and the app.

Geometry comes from geoBoundaries ADM1, which for Moldova is the raion level.
Downloaded once into data/raw and cached.
"""
from __future__ import annotations
import json
import urllib.request

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from utils import RAW

GEOJSON_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
               "releaseData/gbOpen/MDA/ADM1/geoBoundaries-MDA-ADM1_simplified.geojson")

# geoBoundaries spells three of them differently from the statistics office.
ALIASES = {"Municipiul  Balti": "Balti", "Riscani": "RIscani",
           "Singerei": "SIngerei", "Municipiul Chisinau": "Chisinau",
           "U.T.A Gagauzia": "Gagauzia"}


def ensure_geometry() -> dict:
    path = RAW / "raions.geojson"
    if not path.exists():
        with urllib.request.urlopen(GEOJSON_URL, timeout=120) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def _rings(geometry: dict) -> list[np.ndarray]:
    if geometry["type"] == "Polygon":
        parts = [geometry["coordinates"]]
    else:
        parts = geometry["coordinates"]
    return [np.asarray(poly[0], dtype=float) for poly in parts]


def choropleth(ax, values: dict[str, float], cmap: str = "RdYlGn",
               vmin: float | None = None, vmax: float | None = None,
               label: str = "") -> None:
    """Fill each raion by its value; raions without data stay grey."""
    shapes = ensure_geometry()
    lookup = {f["properties"]["shapeName"]: f["geometry"] for f in shapes["features"]}

    vals = np.array(list(values.values()), dtype=float)
    vmin = float(np.nanmin(vals)) if vmin is None else vmin
    vmax = float(np.nanmax(vals)) if vmax is None else vmax
    span = vmax - vmin or 1.0

    patches, colours, blanks = [], [], []
    for name, geometry in lookup.items():
        source = next((k for k, v in ALIASES.items() if v == name), name)
        value = values.get(source, values.get(name))
        for ring in _rings(geometry):
            patch = Polygon(ring, closed=True)
            if value is None or not np.isfinite(value):
                blanks.append(patch)
            else:
                patches.append(patch)
                colours.append((value - vmin) / span)

    if blanks:
        ax.add_collection(PatchCollection(blanks, facecolor="#e8e8e8",
                                          edgecolor="white", linewidths=0.5))
    filled = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidths=0.5)
    filled.set_array(np.asarray(colours))
    filled.set_clim(0, 1)
    ax.add_collection(filled)
    ax.autoscale_view()
    ax.set_aspect(1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    if label:
        ax.set_title(label)
