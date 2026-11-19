#!/usr/bin/env python3
"""Download official PE samples, sky maps, and ZTF flare light curves.

Run with uv (preferred):

    uv run python scripts/fetch_data.py --help
    uv run python scripts/fetch_data.py pe --events GW190521 GW231123
    uv run python scripts/fetch_data.py skymaps --events GW190521 GW231123
    uv run python scripts/fetch_data.py ztf --object ZTF19abanrhr
    uv run python scripts/fetch_data.py all --events GW190521 GW231123

All outputs land under ``data/`` (gitignored for raw bulk files).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PE_DIR = DATA / "pe"
SKY_DIR = DATA / "skymaps"
ZTF_DIR = DATA / "ztf"

# Official bulk releases (GWTC-5.0 / O4b). Prefer per-event GWOSC links when possible.
GWTC5_PE_ZENODO = ("20348005", "20348006")  # PE samples, split across two records
GWTC5_SKY_ZENODO = "20276105"  # archived multi-order / PE sky maps
GWTC5_SKY_TARBALL = "IGWN-GWTC5p0-29ebe06b7_25-Archived_Skymaps.tar.gz"

# Older catalogs needed for GW190521 (O3).
GWTC21_SKY_ZENODO = "6513631"
GWTC21_SKY_TARBALL = "IGWN-GWTC2p1-v2-PESkyMaps.tar.gz"

# AGN J124942.3+344929 / ZTF19abanrhr (Graham et al. 2020)
ZTF19ABANRHR = {
    "object": "ZTF19abanrhr",
    "ra": 192.426223,
    "dec": 34.824839,
    "radius_deg": 0.002,  # ~7 arcsec
}


def _download(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        print(f"  exists, skip: {dest}")
        return dest
    print(f"  GET {url}")
    print(f"  ->  {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


GWOSC_CATALOGS = [
    "GWTC-5.0",
    "GWTC-4.0",
    "GWTC-3-confident",
    "GWTC-2.1-confident",
    "GWTC-1-confident",
]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _gwosc_event_json(event: str) -> tuple[str, dict]:
    """Return (full event name, GWOSC Event API record) for a confident catalog event.

    Accepts full names (GW231123_135430) and short names (GW190521, GW231123).
    """
    last_err: Exception | None = None
    # The Event API accepts unversioned names and redirects to the latest version.
    for cat in GWOSC_CATALOGS:
        try:
            events = _get_json(f"https://gwosc.org/eventapi/json/{cat}/{event}/")["events"]
        except Exception as exc:  # noqa: BLE001 — try next catalog
            last_err = exc
            continue
        name, record = next(iter(events.items()))
        return name.rsplit("-v", 1)[0], record
    # O4 events are only listed under their full name, so search each catalog for EVENT_*.
    for cat in GWOSC_CATALOGS:
        try:
            events = _get_json(f"https://gwosc.org/eventapi/json/{cat}/")["events"]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        matches = sorted({rec["commonName"] for rec in events.values() if rec["commonName"].startswith(f"{event}_")})
        if len(matches) > 1:
            raise RuntimeError(f"{event} is ambiguous in {cat}: {matches}. Pass the full name.")
        if matches:
            return _gwosc_event_json(matches[0])
    raise RuntimeError(f"Could not resolve {event} on GWOSC Event API") from last_err


def _preferred_pe(record: dict) -> dict | None:
    """Return the preferred PE entry of a GWOSC event record; its data_url is the PE data release."""
    pe = [p for p in record.get("parameters", {}).values() if p.get("pipeline_type") == "pe" and p.get("data_url")]
    preferred = [p for p in pe if p.get("is_preferred")]
    return (preferred or pe or [None])[0]


def _skymap_tarball(pe: dict) -> tuple[str, str]:
    """Return (file name, URL) of the archived sky-map tarball that goes with a PE data release."""
    # GWTC-2.1 and GWTC-3 records link the tarball directly (GWTC-3 calls it PESkyLocalizations).
    link = pe.get("links", {}).get("skymap", "")
    match = re.search(r"/files/([^/]+\.tar\.gz)", link)
    if match:
        return urllib.parse.unquote(match.group(1)), link
    # GWTC-4.0 records have no sky-map link, so look next to the PE file in its Zenodo record.
    record_id = re.search(r"/records/(\d+)/", pe["data_url"]).group(1)
    for f in _get_json(f"https://zenodo.org/api/records/{record_id}")["files"]:
        key = f["key"].lower()
        if ("skymap" in key or "skylocalization" in key) and key.endswith(".tar.gz"):
            return f["key"], f"https://zenodo.org/records/{record_id}/files/{urllib.parse.quote(f['key'])}?download=1"
    raise RuntimeError(f"No sky-map tarball in Zenodo record {record_id}")


def _pick_skymap(paths: list[Path], waveform: str) -> Path:
    """Prefer the sky map of the preferred PE waveform, then Mixed, then IMRPhenomXPHM."""
    # GWOSC labels look like "C00:Mixed+XO4a"; file names carry only "Mixed+XO4a".
    for key in (waveform.split(":")[-1], "Mixed", "IMRPhenomXPHM"):
        token = re.compile(rf"(?<![A-Za-z0-9+]){re.escape(key)}(?![A-Za-z0-9+])")
        hits = [p for p in paths if key and token.search(p.name)]
        if hits:
            return hits[0]
    return paths[0]


def fetch_pe_gwosc(events: list[str], overwrite: bool = False) -> list[Path]:
    """Download the preferred official PE data release for each event, as linked from GWOSC."""
    saved: list[Path] = []
    for event in events:
        print(f"[pe] {event}")
        _, record = _gwosc_event_json(event)
        pe = _preferred_pe(record)
        if pe is None:
            print("  WARNING: no PE data release linked from GWOSC for this event.")
            print("  Use Zenodo bulk PE releases instead (see docs/git-prep.md).")
            continue
        print(f"  preferred PE: {pe.get('waveform_family')}")
        dest = PE_DIR / f"{event}_PEDataRelease.hdf5"
        saved.append(_download(pe["data_url"], dest, overwrite=overwrite))
    return saved


def fetch_skymaps_gwosc(events: list[str], overwrite: bool = False) -> list[Path]:
    """Save the official PE sky map of each event as data/skymaps/<event>_skymap.fits.

    Sky maps ship as one tarball per catalog, in the same Zenodo record as the PE data
    release. Only this event's FITS files are extracted from it.
    """
    saved: list[Path] = []
    for event in events:
        print(f"[skymap] {event}")
        dest = SKY_DIR / f"{event}_skymap.fits"
        if dest.exists() and not overwrite:
            print(f"  exists, skip: {dest}")
            saved.append(dest)
            continue
        full_name, record = _gwosc_event_json(event)
        pe = _preferred_pe(record)
        if pe is None:
            print("  WARNING: no PE data release linked from GWOSC for this event.")
            print("  Use Zenodo archived skymap tarballs instead (see docs/git-prep.md).")
            continue
        # Reuse FITS files already extracted, e.g. by an earlier --bulk run.
        # GWTC-2.1 ships plain .fits, GWTC-4.0 ships .fits.gz.
        candidates = [p for p in sorted(SKY_DIR.rglob(f"*{full_name}*.fits*")) if p != dest]
        if not candidates:
            name, url = _skymap_tarball(pe)
            tarball = _download(url, SKY_DIR / name)
            extract_dir = SKY_DIR / name.removesuffix(".tar.gz")
            print(f"  extracting {full_name} -> {extract_dir}")
            with tarfile.open(tarball, "r:gz") as tar:
                members = [m for m in tar.getmembers() if full_name in m.name and m.name.endswith((".fits", ".fits.gz"))]
                tar.extractall(extract_dir, members=members, filter="data")
            candidates = sorted(extract_dir.rglob(f"*{full_name}*.fits*"))
        if not candidates:
            print(f"  WARNING: no sky map for {full_name} in the archived tarball.")
            continue
        chosen = _pick_skymap(candidates, pe.get("waveform_family", ""))
        print(f"  using {chosen.name}")
        if chosen.suffix == ".gz":
            with gzip.open(chosen, "rb") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        else:
            shutil.copyfile(chosen, dest)
        saved.append(dest)
    return saved


def fetch_zenodo_record(record_id: str, dest_dir: Path, files: list[str] | None = None) -> None:
    """Download a Zenodo record with zenodo_get (must be installed in the uv env)."""
    try:
        from zenodo_get import zget  # type: ignore
    except ImportError:
        # zenodo-get exposes a CLI; call it as a module fallback
        import subprocess

        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "zenodo_get", record_id, "-o", str(dest_dir)]
        if files:
            for f in files:
                cmd.extend(["-g", f])
        print("  running:", " ".join(cmd))
        subprocess.check_call(cmd)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Library API varies by version; prefer CLI path above when unsure.
    print(f"  zenodo_get {record_id} -> {dest_dir}")
    zget([record_id, "-o", str(dest_dir)])


def fetch_skymaps_zenodo_bulk(catalog: str = "gwtc5") -> Path:
    """Download archived sky-map tarball from the official Zenodo data release."""
    SKY_DIR.mkdir(parents=True, exist_ok=True)
    if catalog == "gwtc5":
        record, name = GWTC5_SKY_ZENODO, GWTC5_SKY_TARBALL
    elif catalog == "gwtc2.1":
        record, name = GWTC21_SKY_ZENODO, GWTC21_SKY_TARBALL
    else:
        raise ValueError(catalog)

    url = f"https://zenodo.org/records/{record}/files/{urllib.parse.quote(name)}?download=1"
    dest = SKY_DIR / name
    _download(url, dest)
    extract_dir = SKY_DIR / catalog
    if not extract_dir.exists():
        print(f"  extracting -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(extract_dir)
    return extract_dir


def fetch_ztf_irsa(
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    object_name: str = "ZTF19abanrhr",
    overwrite: bool = False,
) -> Path:
    """Download ZTF light curves from the IRSA ZTF Light Curve API (official).

    Docs: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
    """
    ra = ZTF19ABANRHR["ra"] if ra is None else ra
    dec = ZTF19ABANRHR["dec"] if dec is None else dec
    radius_deg = ZTF19ABANRHR["radius_deg"] if radius_deg is None else radius_deg

    # POS=CIRCLE RA DEC RADIUS (degrees). BAD_CATFLAGS_MASK=32768 is a common quality cut.
    query = urllib.parse.urlencode(
        {
            "POS": f"CIRCLE {ra} {dec} {radius_deg}",
            "BAD_CATFLAGS_MASK": "32768",
            "FORMAT": "csv",
        }
    )
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{query}"
    dest = ZTF_DIR / f"{object_name}_irsa.csv"
    _download(url, dest, overwrite=overwrite)

    # Optional public broker page (Lasair ZTF).
    lasair = ZTF_DIR / f"{object_name}_lasair.html"
    try:
        _download(
            f"https://lasair-ztf.lsst.ac.uk/objects/{object_name}/",
            lasair,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (optional Lasair page skipped: {exc})")

    meta = ZTF_DIR / f"{object_name}_meta.json"
    meta.write_text(
        json.dumps(
            {
                "object": object_name,
                "ra": ra,
                "dec": dec,
                "radius_deg": radius_deg,
                "host_agn": "J124942.3+344929",
                "source": "IRSA ZTF light-curve API",
                "api_docs": "https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html",
                "csv": str(dest.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {meta}")
    return dest


def fetch_ztf_ztfquery(object_name: str = "ZTF19abanrhr") -> Path:
    """Optional: same light curve via ztfquery (IRSA under the hood)."""
    from ztfquery import lightcurve

    ra, dec, radius = (
        ZTF19ABANRHR["ra"],
        ZTF19ABANRHR["dec"],
        ZTF19ABANRHR["radius_deg"] * 3600.0,  # arcsec
    )
    print(f"[ztf/ztfquery] {object_name} @ ({ra}, {dec})")
    lcq = lightcurve.LCQuery.from_position(ra, dec, radius)
    dest = ZTF_DIR / f"{object_name}_ztfquery.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    lcq.data.to_csv(dest, index=False)
    print(f"  -> {dest}")
    return dest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("pe", help="Official PE HDF5 via GWOSC Event API")
    pe.add_argument("--events", nargs="+", default=["GW190521", "GW231123"])
    pe.add_argument("--overwrite", action="store_true")

    sky = sub.add_parser("skymaps", help="Official PE sky-map FITS via GWOSC Event API")
    sky.add_argument("--events", nargs="+", default=["GW190521", "GW231123"])
    sky.add_argument("--overwrite", action="store_true")
    sky.add_argument(
        "--bulk",
        choices=["none", "gwtc5", "gwtc2.1"],
        default="none",
        help="Also download Zenodo archived skymap tarball",
    )

    ztf = sub.add_parser("ztf", help="ZTF flare light curves from IRSA")
    ztf.add_argument("--object", default="ZTF19abanrhr")
    ztf.add_argument("--ra", type=float, default=None)
    ztf.add_argument("--dec", type=float, default=None)
    ztf.add_argument("--radius-deg", type=float, default=None)
    ztf.add_argument("--ztfquery", action="store_true", help="Also fetch via ztfquery")
    ztf.add_argument("--overwrite", action="store_true")

    bulk = sub.add_parser("zenodo-pe", help="Bulk GWTC-5.0 PE from Zenodo (large!)")
    bulk.add_argument("--records", nargs="+", default=list(GWTC5_PE_ZENODO))

    all_p = sub.add_parser("all", help="PE + skymaps (GWOSC) + ZTF19abanrhr (IRSA)")
    all_p.add_argument("--events", nargs="+", default=["GW190521", "GW231123"])
    all_p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)

    if args.cmd == "pe":
        fetch_pe_gwosc(args.events, overwrite=args.overwrite)
    elif args.cmd == "skymaps":
        fetch_skymaps_gwosc(args.events, overwrite=args.overwrite)
        if args.bulk != "none":
            fetch_skymaps_zenodo_bulk(args.bulk)
    elif args.cmd == "ztf":
        fetch_ztf_irsa(
            ra=args.ra,
            dec=args.dec,
            radius_deg=args.radius_deg,
            object_name=args.object,
            overwrite=args.overwrite,
        )
        if args.ztfquery:
            fetch_ztf_ztfquery(args.object)
    elif args.cmd == "zenodo-pe":
        for rec in args.records:
            fetch_zenodo_record(rec, PE_DIR / f"zenodo_{rec}")
    elif args.cmd == "all":
        fetch_pe_gwosc(args.events, overwrite=args.overwrite)
        fetch_skymaps_gwosc(args.events, overwrite=args.overwrite)
        fetch_ztf_irsa(object_name="ZTF19abanrhr", overwrite=args.overwrite)
    else:
        raise SystemExit(f"unknown command {args.cmd}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
