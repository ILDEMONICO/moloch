#!/usr/bin/env python3
"""
fetch_moloch.py

Gira dentro GitHub Actions. Scarica i file GRIB2 di MOLOCH-AIM direttamente
da /nwp/MOLOCH_AIM/ su MeteoHub (niente bundle API, che non risulta
funzionante per questo dataset), scopre da solo i nomi file scorrendo
l'index Apache delle cartelle, ed estrae con ecCodes il valore piu' vicino
a Catania per ogni messaggio (= ogni scadenza temporale) nel file.

Uso:
    python scripts/fetch_moloch.py --out data/dashboard.json
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import eccodes as ec

BASE = "https://meteohub.agenziaitaliameteo.it/nwp/MOLOCH_AIM"

DEFAULT_LAT = 37.5079
DEFAULT_LON = 15.0830
PLACE_NAME = "Catania"

# variabile amichevole -> nome cartella su /nwp/MOLOCH_AIM/{run}/
VAR_FOLDERS = {
    "temperatura": "2t",
    "vento_u": "10u",
    "vento_v": "10v",
    "umidita": "2r",
}
PRECIP_FOLDER = "unknown"  # da verificare via log — vedi discovered_shortnames


def list_index(url: str):
    """Ritorna gli href elencati in una pagina Apache 'Index of'."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', r.text)
    return [h for h in hrefs if h not in ("../",)]


def find_latest_run() -> str:
    """Trova la cartella run piu' recente sotto MOLOCH_AIM/ (formato YYYYMMDDHH)."""
    entries = list_index(BASE + "/")
    runs = sorted(e.strip("/") for e in entries if re.match(r"^\d{10}/$", e))
    if not runs:
        raise RuntimeError(f"Nessuna cartella run trovata sotto {BASE}/")
    return runs[-1]


def find_grib_file(run: str, folder: str) -> str:
    """Trova il (primo/unico) file .grib in una cartella variabile."""
    url = f"{BASE}/{run}/{folder}/"
    entries = list_index(url)
    gribs = [e for e in entries if e.endswith(".grib")]
    if not gribs:
        raise RuntimeError(f"Nessun file .grib trovato in {url}")
    return url + gribs[0]


def download(url: str, out_path: Path):
    print(f"Scarico {url} ...", file=sys.stderr)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def extract_nearest_series(grib_path: Path, lat: float, lon: float, shortname_filter=None):
    """
    Scorre tutti i messaggi GRIB nel file, estrae il valore piu' vicino a
    (lat, lon) per ognuno. Ritorna lista di dict {valid_time, shortname, value}
    e l'insieme di tutti gli shortName incontrati (utile per debug).
    """
    out = []
    seen_shortnames = set()
    with open(grib_path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                short_name = ec.codes_get(gid, "shortName")
                seen_shortnames.add(short_name)
                if shortname_filter and short_name not in shortname_filter:
                    continue

                valid_date = ec.codes_get(gid, "validityDate")  # YYYYMMDD
                valid_time = ec.codes_get(gid, "validityTime")  # HMM or HHMM
                dt = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")

                nearest = ec.codes_grib_find_nearest(gid, lat, lon)
                best = min(nearest, key=lambda p: p.distance)

                out.append({"valid_time": dt, "shortname": short_name, "value": best.value})
            finally:
                ec.codes_release(gid)
    return out, seen_shortnames


def to_series_dict(entries):
    """Converte lista di {valid_time, value} (gia' filtrata per 1 variabile) in due liste allineate."""
    entries = sorted(entries, key=lambda e: e["valid_time"])
    times = [e["valid_time"] for e in entries]
    values = [e["value"] for e in entries]
    return times, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--place-name", default=PLACE_NAME)
    ap.add_argument("--workdir", default="./moloch_tmp")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    run = find_latest_run()
    print(f"Run piu' recente trovato: {run}", file=sys.stderr)

    series = {}
    master_times = None

    for friendly, folder in VAR_FOLDERS.items():
        file_url = find_grib_file(run, folder)
        local = workdir / f"{folder}.grib"
        download(file_url, local)
        entries, shortnames = extract_nearest_series(local, args.lat, args.lon)
        print(f"  shortName trovati in {folder}: {shortnames}", file=sys.stderr)
        times, values = to_series_dict(entries)
        series[friendly] = values
        if master_times is None:
            master_times = times
        local.unlink(missing_ok=True)

    # Precipitazione: cartella "unknown", shortName da scoprire dal log.
    precip_url = find_grib_file(run, PRECIP_FOLDER)
    local = workdir / "precip.grib"
    download(precip_url, local)
    entries, shortnames = extract_nearest_series(local, args.lat, args.lon)
    print(f"  shortName trovati in {PRECIP_FOLDER}: {shortnames}", file=sys.stderr)
    # Proviamo gli shortName piu' comuni per la precipitazione totale.
    precip_candidates = ["tp", "prate", "tirf", "rain", "acraint"]
    chosen = next((s for s in precip_candidates if s in shortnames), None)
    if chosen:
        print(f"  Uso '{chosen}' come precipitazione.", file=sys.stderr)
        filtered = [e for e in entries if e["shortname"] == chosen]
        _, precip_values = to_series_dict(filtered)
    else:
        print(f"  ATTENZIONE: nessuno shortName noto per la precipitazione tra {shortnames}. "
              f"Serie lasciata vuota, segnalarlo per correggere lo script.", file=sys.stderr)
        precip_values = []
    local.unlink(missing_ok=True)

    # Temperatura: Kelvin -> Celsius se necessario
    temp_c = [round(v - 273.15, 1) if v > 100 else round(v, 1) for v in series["temperatura"]]

    # Vento: modulo da u/v
    vento = [round((u ** 2 + v ** 2) ** 0.5, 1) for u, v in zip(series["vento_u"], series["vento_v"])]

    umidita = [round(v, 0) for v in series["umidita"]]

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "place": args.place_name,
        "run": run,
        "labels": [t.strftime("%d/%m %H:%M") for t in master_times],
        "temperatura": temp_c,
        "vento": vento,
        "umidita": umidita,
        "precipitazione": [round(v, 2) for v in precip_values] if precip_values else [],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scritto {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
