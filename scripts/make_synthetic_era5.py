"""Write a small synthetic dataset in the ERA5 HDF5 layout the loaders expect.

Each channel is a sum of a few eastward-travelling waves plus noise, so the
next step is predictable and the loss drops during training. The layout matches
latlon_025deg_hdf5_1h on a coarser grid:

    <root>/<name>/<year>.h5           data [time, channel, lat, lon], time, channel, latitude, longitude
    <root>/<name>/stats/stats_<tag>.h5 global_mean, global_std per channel
    <root>/<name>/invar/invariants.h5  orog, lsm

Example (1 degree, 6-hourly, 90 training days in 2016, 20 validation days in 2017):

    python scripts/make_synthetic_era5.py --root /path/to/data --years 2016:90 2017:20
"""

import argparse
import os
from datetime import datetime

import h5py
import numpy as np

SFC_VARS = ["u10m", "v10m", "t2m", "tcwv", "sp", "msl"]
PL_VARS = ["u", "v", "z", "t", "q"]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CHANNELS = SFC_VARS + [f"{v}{lev}" for v in PL_VARS for lev in LEVELS]

# (zonal wavenumber, period in hours, centre latitude) for each wave
WAVES = [(1, 240.0, 45.0), (2, 120.0, -30.0), (3, 72.0, 0.0), (5, 48.0, 60.0)]
NOISE = 0.05


def hours_since_1900(year: int) -> float:
    return (datetime(year, 1, 1) - datetime(1900, 1, 1)).total_seconds() / 3600.0


def wave_basis(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """[wave, cos/sin, lat, lon] spatial patterns."""
    lon_rad = np.deg2rad(lons)
    basis = []
    for m, _, lat0 in WAVES:
        env = np.exp(-(((lats - lat0) / 30.0) ** 2))[:, None]
        basis.append([env * np.cos(m * lon_rad)[None], env * np.sin(m * lon_rad)[None]])
    return np.asarray(basis, dtype=np.float32)


def write_year(path, year, n_days, dt_hours, lats, lons, basis, amps, phases, rng):
    n_times = int(n_days * 24 // dt_hours)
    t0 = hours_since_1900(year)
    times = t0 + dt_hours * np.arange(n_times, dtype=np.float64)
    omegas = np.array([2 * np.pi / period for _, period, _ in WAVES])
    shape = (n_times, len(CHANNELS), len(lats), len(lons))
    with h5py.File(path, "w") as f:
        f.create_dataset("channel", data=np.array(CHANNELS, dtype="S5"))
        f.create_dataset("latitude", data=lats.astype(np.float32))
        f.create_dataset("longitude", data=lons.astype(np.float32))
        f.create_dataset("time", data=times)
        data = f.create_dataset("data", shape=shape, dtype=np.float32)
        for i, t in enumerate(times):
            # cos(m*lon - w*t + p) = cos(m*lon) cos(w*t - p) + sin(m*lon) sin(w*t - p)
            arg = omegas[None, :] * t - phases  # [channel, wave]
            coef = amps[..., None] * np.stack([np.cos(arg), np.sin(arg)], axis=-1)
            field = np.einsum("ckj,kjyx->cyx", coef.astype(np.float32), basis)
            field += NOISE * rng.standard_normal(field.shape, dtype=np.float32)
            data[i] = field
    print(f"wrote {path}: {shape}")


def write_stats(path, tag, year_files):
    # Mean and std per channel over a sample of time steps
    sums, sq_sums, count = 0.0, 0.0, 0
    for yf in year_files:
        with h5py.File(yf, "r") as f:
            idx = np.unique(np.linspace(0, f["data"].shape[0] - 1, 16).astype(int))
            sample = f["data"][idx].astype(np.float64)
        sums = sums + sample.sum(axis=(0, 2, 3))
        sq_sums = sq_sums + (sample**2).sum(axis=(0, 2, 3))
        count += sample.shape[0] * sample.shape[2] * sample.shape[3]
    mean = sums / count
    std = np.sqrt(sq_sums / count - mean**2)
    with h5py.File(path, "w") as f:
        f.create_dataset("channel", data=np.array(CHANNELS, dtype="S5"))
        f.create_dataset("global_mean", data=mean.astype(np.float32))
        f.create_dataset("global_std", data=std.astype(np.float32))
        f.create_dataset("temp_diff_std", data=std.astype(np.float32))
        f.attrs["tag"] = tag
    print(f"wrote {path}")


def write_invariants(path, lats, lons, basis, rng):
    orog = np.clip(basis[:, 0].sum(axis=0) + 0.3 * rng.standard_normal(), 0, None) * 1000.0
    lsm = (orog > 0).astype(np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("channel", data=np.array(["orog", "lsm"], dtype="S4"))
        f.create_dataset("data", data=np.stack([orog, lsm]).astype(np.float64))
        f.create_dataset("latitude", data=lats.astype(np.float64))
        f.create_dataset("longitude", data=lons.astype(np.float64))
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="data root (cfg.data.root)")
    parser.add_argument("--name", default="latlon_1deg_synthetic_6h", help="dataset folder (cfg.data.name)")
    parser.add_argument("--tag", default="v1.0", help="stats tag (cfg.data.tag)")
    parser.add_argument("--resolution", type=float, default=1.0, help="grid spacing in degrees")
    parser.add_argument("--dt-hours", type=int, default=6, help="hours between time steps")
    parser.add_argument("--years", nargs="+", default=["2016:90", "2017:20"], help="YEAR:DAYS entries")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_lat = int(round(180 / args.resolution)) + 1  # includes both poles; loaders drop the last row
    n_lon = int(round(360 / args.resolution))
    lats = np.linspace(90, -90, n_lat)
    lons = np.arange(n_lon) * args.resolution
    basis = wave_basis(lats, lons)
    amps = rng.standard_normal((len(CHANNELS), len(WAVES))) / np.sqrt(len(WAVES))
    phases = rng.uniform(0, 2 * np.pi, (len(CHANNELS), len(WAVES)))

    out = os.path.join(args.root, args.name)
    os.makedirs(os.path.join(out, "stats"), exist_ok=True)
    os.makedirs(os.path.join(out, "invar"), exist_ok=True)
    year_files = []
    for entry in args.years:
        year, days = (int(x) for x in entry.split(":"))
        path = os.path.join(out, f"{year}.h5")
        write_year(path, year, days, args.dt_hours, lats, lons, basis, amps, phases, rng)
        year_files.append(path)
    write_stats(os.path.join(out, "stats", f"stats_{args.tag}.h5"), args.tag, year_files)
    write_invariants(os.path.join(out, "invar", "invariants.h5"), lats, lons, basis, rng)


if __name__ == "__main__":
    main()
