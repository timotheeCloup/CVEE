"""Generate ``api/communes_coords.json`` from the official French geo dataset.

Source: https://geo.api.gouv.fr (data.gouv.fr, Licence Ouverte, free, no API key).
Two endpoints are merged: all communes and the 45 municipal arrondissements
(Paris / Lyon / Marseille), whose coordinates are more precise than their
parent city.

The output is a compact lookup used by the API to place offers that carry no
latitude/longitude on the map:

    {"insee": {code -> [lat, lon]}, "cp": {postal -> [lat, lon]},
     "dept": {department -> [lat, lon]}}

``dept`` is the mean of the commune centroids of each department, used as a
last-resort fallback when only a department is known ("13 - Bouches-du-Rhône").

Usage:
    uv run python scripts/build_communes_coords.py
"""

import json
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://geo.api.gouv.fr/communes"
FIELDS = "nom,code,codesPostaux,centre"
OUTPUT = Path(__file__).resolve().parent.parent / "api" / "communes_coords.json"
USER_AGENT = "CVEE-build-communes-coords/1.0"


def department_of(insee: str) -> str:
    """Return the department code for an INSEE commune code (2A/2B, 3-digit overseas)."""
    return insee[:3] if insee.startswith("97") else insee[:2]


def fetch(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def build() -> dict[str, dict[str, list[float]]]:
    communes = fetch(f"{BASE_URL}?fields={FIELDS}&format=json")
    arrondissements = fetch(f"{BASE_URL}?type=arrondissement-municipal&fields={FIELDS}&format=json")

    insee_map: dict[str, list[float]] = {}
    cp_map: dict[str, list[float]] = {}

    # Arrondissements first so their precise coordinates win the shared
    # postal-code keys over the parent city.
    for commune in arrondissements + communes:
        centre = commune.get("centre")
        if not centre:
            continue
        lon, lat = centre["coordinates"]
        coords = [round(float(lat), 4), round(float(lon), 4)]
        insee_map.setdefault(commune["code"], coords)
        for postal_code in commune.get("codesPostaux") or []:
            cp_map.setdefault(postal_code, coords)

    totals: dict[str, list[float]] = {}
    for insee, coords in insee_map.items():
        acc = totals.setdefault(department_of(insee), [0.0, 0.0, 0.0])
        acc[0] += coords[0]
        acc[1] += coords[1]
        acc[2] += 1
    dept_map = {
        dept: [round(acc[0] / acc[2], 4), round(acc[1] / acc[2], 4)] for dept, acc in totals.items()
    }

    return {"insee": insee_map, "cp": cp_map, "dept": dept_map}


def main() -> None:
    data = build()
    OUTPUT.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print({key: len(value) for key, value in data.items()})


if __name__ == "__main__":
    main()
