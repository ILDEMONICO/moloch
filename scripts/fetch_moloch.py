#!/usr/bin/env python3
"""
fetch_moloch.py

Pensato per girare dentro GitHub Actions (non sul tuo hosting).
Scarica l'ultimo run disponibile di MOLOCH-AIM da MeteoHub, estrae la
serie temporale nel punto piu' vicino a Catania e salva un JSON compatto
che la pagina HTML su Altervista andra' a leggere via jsDelivr.

Uso (dentro il workflow):
    python scripts/fetch_moloch.py --out data/dashboard.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import xarray as xr
import pandas as pd

BASE_URL = "https://meteohub.agenziaitaliameteo.it"
DATASET_ID = "MOLOCH-AIM"  # verifica il valore esatto su /app/datasets se questo non funziona

DEFAULT_LAT = 37.5079
DEFAULT_LON = 15.0830
PLACE_NAME = "Catania"

VARIABLES = {
    "temperatura": ["t2m", "2t", "temperature_2m"],
    "precipitazione": ["tp", "precipitation", "total_precipitation"],
    "vento_u": ["u10", "10u"],
    "vento_v": ["v10", "10v"],
    "umidita": ["r2", "rh2m", "relative_humidity_2m", "2r"],
}

# I run MOLOCH tipicamente disponibili: 00 e 12 UTC. Proviamo le combinazioni
# piu' recenti a scalare finche' una scarica con successo.
def candidate_reftimes(n=6):
    now = datetime.now(timezone.utc)
    candidates = []
    day = now
    runs = ["12", "00"]
    # partiamo dall'ultimo run gia' passato
    for i in range(n):
        for run in runs:
            reftime = day.strftime("%Y%m%d")
            run_dt = day.replace(hour=int(run), minute=0, second=0, microsecond=0)
            if run_dt <= now:
                candidates.append((reftime, f"{run}:00"))
        day -= timedelta(days=1)
    return candidates


def try_download(dataset_id: str, reftime: str, run: str, out_path: Path) -> bool:
    url = f"{BASE_URL}/api/opendata/{dataset_id}/download"
    params = {"reftime": reftime, "run": run}
    try:
        with requests.get(url, params=params, stream=True, timeout=120) as r:
            if r.status_code != 200:
                return False
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return out_path.stat().st_size > 1000  # scarta risposte vuote/errore mascherate da 200
    except requests.RequestException:
        return False


def find_var(ds, aliases):
    for name in aliases:
        if name in ds.data_vars:
            return ds[name]
    return None


def extract_series(grib_path: Path, lat: float, lon: float) -> pd.DataFrame:
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    print("Variabili nel file:", list(ds.data_vars), file=sys.stderr)

    records = {}
    for friendly_name, aliases in VARIABLES.items():
        var = find_var(ds, aliases)
        if var is None:
            continue
        series = var.sel(latitude=lat, longitude=lon, method="nearest")
        records[friendly_name] = series.values.tolist()

    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    if "valid_time" in point.coords:
        times = pd.to_datetime(point["valid_time"].values)
    elif "step" in point.coords:
        times = pd.to_datetime(point["step"].values)
    else:
        times = pd.to_datetime(point["time"].values)

    df = pd.DataFrame(records)
    df.insert(0, "datetime", times)

    if "vento_u" in df.columns and "vento_v" in df.columns:
        df["vento"] = (df["vento_u"] ** 2 + df["vento_v"] ** 2) ** 0.5
        df.drop(columns=["vento_u", "vento_v"], inplace=True)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path del file JSON di output")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--place-name", default=PLACE_NAME)
    args = ap.parse_args()

    tmp_grib = Path("bundle.grib2")
    used_reftime, used_run = None, None

    for reftime, run in candidate_reftimes():
        print(f"Provo reftime={reftime} run={run}...", file=sys.stderr)
        if try_download(args.dataset_id, reftime, run, tmp_grib):
            used_reftime, used_run = reftime, run
            print(f"OK: run {reftime} {run} disponibile.", file=sys.stderr)
            break
    else:
        print("Nessun run disponibile trovato tra i candidati.", file=sys.stderr)
        sys.exit(1)

    df = extract_series(tmp_grib, args.lat, args.lon)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "place": args.place_name,
        "reftime": used_reftime,
        "run": used_run,
        "labels": [t.strftime("%d/%m %H:%M") for t in df["datetime"]],
        "temperatura": df.get("temperatura", pd.Series(dtype=float)).round(1).tolist(),
        "precipitazione": df.get("precipitazione", pd.Series(dtype=float)).round(2).tolist(),
        "vento": df.get("vento", pd.Series(dtype=float)).round(1).tolist(),
        "umidita": df.get("umidita", pd.Series(dtype=float)).round(0).tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scritto {out_path}", file=sys.stderr)

    tmp_grib.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
