import json, math, os, re, shutil
from datetime import datetime, timezone
from urllib.parse import urljoin
from zoneinfo import ZoneInfo
import numpy as np
import requests
import eccodes as ec

RAIN_MODE = "interval"      # MOLOCH
ROOT = "https://meteohub.agenziaitaliameteo.it/nwp/MOLOCH_AIM/"
LAT, LON = 37.521499, 15.084048
RADIUS_KM = 10          # intorno usato per la "probabilità" di pioggia
RAIN_MM_H = 0.2         # soglia pioggia oraria
LATEST = "latest.json"
HIST_DIR = "storico"
TZNAME = "Europe/Rome"
TZ = ZoneInfo(TZNAME)
HEADERS = {"User-Agent": "moloch-personal-forecast (tua@email)"}

# job -> lista di (cartella, (discipline, categoria, numero))
JOBS = {
    "temp":  [("2t", (0, 0, 0))],
    "rh":    [("2r", (0, 1, 1))],
    "wind":  [("10u", (0, 2, 2)), ("10v", (0, 2, 3))],
    "gust":  [("gust", (0, 2, 22))],
    "cloud": [("clct", (0, 6, 1))],
    "rain":  [("unknown", (0, 1, 8))],   # unknown = tp cumulata
}
FOLDERS = [f for j in JOBS.values() for f, _ in j]
TP = (0, 1, 8)

FIELDS = ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "rain",
          "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
UNITS = {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation_probability": "%",
         "rain": "mm", "cloud_cover": "%", "wind_speed_10m": "km/h",
         "wind_direction_10m": "deg", "wind_gusts_10m": "km/h"}


def hist_name(run):
    """2026102800 -> 'storico/2026-10-28 00-00 moloch.json'"""
    return os.path.join(HIST_DIR, f"{run[:4]}-{run[4:6]}-{run[6:8]} {run[8:10]}-00 moloch.json")


def run_iso(run):
    """2026102800 -> '2026-10-28T00:00:00Z'"""
    return datetime.strptime(run, "%Y%m%d%H").strftime("%Y-%m-%dT%H:%M:00Z")


def links(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out = []
    for href in re.findall(r'href="([^"?#]+)"', r.text):
        full = urljoin(url, href)
        if full.startswith(url) and full != url:
            out.append(full)
    return out


def find_latest_run():
    """Run più recente con tutte le cartelle necessarie già popolate."""
    runs = sorted((u for u in links(ROOT) if re.search(r"/\d{10}/$", u)), reverse=True)
    for run_url in runs:
        try:
            ok = all(any(not x.endswith("/") for x in links(run_url + f + "/")) for f in FOLDERS)
        except Exception:
            ok = False
        if ok:
            return run_url
        print(f"{run_url.split('/')[-2]}: incompleto, provo il precedente")
    raise SystemExit("Nessun run completo trovato")


def g(h, k):
    try:
        return ec.codes_get(h, k)
    except Exception:
        return None


def iter_messages(url):
    """Legge il GRIB in streaming e restituisce un messaggio alla volta."""
    buf = bytearray()
    done = 0
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            buf += chunk
            done += len(chunk)
            print(f"\r  {done/1e6:.0f} MB", end="", flush=True)
            while True:
                i = buf.find(b"GRIB")
                if i < 0:
                    del buf[:-3]
                    break
                if i > 0:
                    del buf[:i]
                if len(buf) < 16:
                    break
                n = int.from_bytes(buf[8:16], "big")
                if len(buf) < n:
                    break
                yield bytes(buf[:n])
                del buf[:n]
    print()


def nearest(h):
    lats = ec.codes_get_array(h, "latitudes")
    lons = (ec.codes_get_array(h, "longitudes") + 180) % 360 - 180
    d = np.hypot((lons - LON) * math.cos(math.radians(LAT)), lats - LAT) * 111.19
    return int(d.argmin()), d, lats, lons


def km(a, b):
    return math.hypot((a[1] - b[1]) * math.cos(math.radians(LAT)), a[0] - b[0]) * 111.19


def get_cell(url):
    """Contorno della cella: legge solo il primo messaggio del file."""
    it = iter_messages(url)
    raw = next(it)
    it.close()
    h = ec.codes_new_from_message(raw)
    idx, d, lats, lons = nearest(h)
    ni, nj = g(h, "Ni"), g(h, "Nj")
    cell = dict(center=[float(lats[idx]), float(lons[idx])], poly=None,
                dist=round(float(d[idx]), 2), grid=g(h, "gridType"))
    if ni and nj and ni * nj == len(lats):
        la, lo = lats.reshape(nj, ni), lons.reshape(nj, ni)
        j, i = divmod(idx, ni)
        poly = []
        for sj, si in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            js = [j, min(max(j + sj, 0), nj - 1)]
            is_ = [i, min(max(i + si, 0), ni - 1)]
            pts = [(la[a, b], lo[a, b]) for a in js for b in is_]
            poly.append([float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))])
        cell["poly"] = poly
        cell["w_km"] = round(km(poly[0], poly[1]), 2)
        cell["h_km"] = round(km(poly[1], poly[2]), 2)
        print(f"  cella {cell['w_km']} x {cell['h_km']} km, centro a {cell['dist']} km dal punto")
    else:
        print("  griglia non regolare: contorno cella non disponibile")
    ec.codes_release(h)
    return cell


def extract(url, key):
    """Serie [(datetime UTC, valore sulla cella, valori nell'intorno o None)]."""
    idx = near = None
    res = []
    for raw in iter_messages(url):
        h = ec.codes_new_from_message(raw)
        k = (g(h, "discipline"), g(h, "parameterCategory"), g(h, "parameterNumber"))
        if k != key:
            ec.codes_release(h)
            continue
        if idx is None:
            idx, d, _, _ = nearest(h)
            near = np.where(d <= RADIUS_KM)[0]
            print(f"  griglia {g(h,'gridType')}, cella a {d[idx]:.1f} km, {len(near)} celle nell'intorno")
            if key[1] == 2:
                print(f"  uvRelativeToGrid = {g(h,'uvRelativeToGrid')}")
        vals = ec.codes_get_values(h)
        t = datetime.strptime(f"{g(h,'validityDate')}{int(g(h,'validityTime')):04d}",
                              "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        res.append((t, float(vals[idx]), vals[near].copy() if key == TP else None))
        ec.codes_release(h)
    res.sort(key=lambda x: x[0])
    return res


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def load(run):
    """Carica il file storico del run se esiste, altrimenti riparte da zero."""
    path = hist_name(run)
    if os.path.exists(path):
        try:
            doc = json.load(open(path, encoding="utf-8"))
            if doc["source_meta"]["run"] == run:
                return doc["source_meta"], {r["time_utc"]: r for r in doc["hourly"]}
        except Exception:
            pass
    return {"run": run, "done": [], "cell": None}, {}


def publish_latest(path):
    tmp = LATEST + ".tmp"
    shutil.copyfile(path, tmp)
    os.replace(tmp, LATEST)


def save(meta, rows):
    os.makedirs(HIST_DIR, exist_ok=True)
    path = hist_name(meta["run"])
    now = datetime.now(timezone.utc)
    hourly = []
    for k in sorted(rows):
        t = datetime.strptime(k, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
        r = {"time_utc": k, "time_local": t.astimezone(TZ).strftime("%Y-%m-%dT%H:%M")}
        r.update({f: rows[k].get(f) for f in FIELDS})
        hourly.append(r)
    doc = {
        "schema_version": 1, "source": "meteohub.agenziaitaliameteo.it", "model": "MOLOCH_AIM",
        "model_run": run_iso(meta["run"]),
        "fetched_at_utc": now.strftime("%Y-%m-%dT%H:%M:00Z"),
        "fetched_at_local": now.astimezone(TZ).strftime("%Y-%m-%dT%H:%M:00%z"),
        "timezone": TZNAME,
        "location": {"latitude": LAT, "longitude": LON},
        "units": UNITS,
        "source_meta": dict(meta, radius_km=RADIUS_KM, rain_threshold_mm_h=RAIN_MM_H,
                            pop_method="% celle entro radius_km con pioggia >= soglia"),
        "hourly": hourly,
    }
    tmp = path + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    publish_latest(path)


def main():
    base = find_latest_run()
    run = base.rstrip("/").split("/")[-1]
    print(f"Run selezionato: {run} -> {hist_name(run)}")
    meta, rows = load(run)

    if meta.get("cell") and all(j in meta["done"] for j in JOBS):
        publish_latest(hist_name(run))
        print("Run già completo: aggiornato solo latest.json")
        return

    def row(t):
        return rows.setdefault(iso(t), {})

    if not meta.get("cell"):
        print("Contorno cella:")
        f0 = [x for x in links(base + "2t/") if not x.endswith("/")][0]
        meta["cell"] = get_cell(f0)
        save(meta, rows)

    for job, parts in JOBS.items():
        if job in meta["done"]:
            print(f"{job}: già presente")
            continue
        series = []
        for folder, key in parts:
            files = [x for x in links(base + folder + "/") if not x.endswith("/")]
            print(f"{job} ({folder}): {files[0].split('/')[-1]}")
            series.append(extract(files[0], key))

        if job == "temp":
            for t, v, _ in series[0]:
                row(t)["temperature_2m"] = round(v - 273.15, 1)
        elif job == "rh":
            for t, v, _ in series[0]:
                row(t)["relative_humidity_2m"] = round(v)
        elif job == "cloud":
            for t, v, _ in series[0]:
                row(t)["cloud_cover"] = round(v)
        elif job == "gust":
            for t, v, _ in series[0]:
                row(t)["wind_gusts_10m"] = round(v * 3.6, 1)
        elif job == "wind":
            vmap = {t: v for t, v, _ in series[1]}
            for t, u, _ in series[0]:
                if t not in vmap:
                    continue
                v = vmap[t]
                r = row(t)
                r["wind_speed_10m"] = round(math.hypot(u, v) * 3.6, 1)
                r["wind_direction_10m"] = round((180 + math.degrees(math.atan2(u, v))) % 360)
        elif job == "rain":
            tp = series[0]
            row(tp[0][0])["rain"] = 0.0
            row(tp[0][0])["precipitation_probability"] = 0
            if RAIN_MODE == "interval":
                for t, v, nv in tp[1:]:
                    r = row(t)
                    r["rain"] = round(max(v, 0), 2)
                    r["precipitation_probability"] = round(float((nv >= RAIN_MM_H).mean()) * 100)
            else:
                for (t0, a, na), (t1, b, nb) in zip(tp, tp[1:]):
                    dt = (t1 - t0).total_seconds() / 3600
                    r = row(t1)
                    r["rain"] = round(max(b - a, 0), 2)
                    r["precipitation_probability"] = round(float(((nb - na) / dt >= RAIN_MM_H).mean()) * 100)

        meta["done"].append(job)
        save(meta, rows)

    print(f"\nFatto: {hist_name(run)} + {LATEST} ({len(rows)} ore)")


main()
