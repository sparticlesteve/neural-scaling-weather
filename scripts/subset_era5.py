"""Copy a time subset of the hourly ERA5 dataset, keeping its layout.

Takes every <stride>-th hour of the first N days of each year. With stride 6,
set data.dt_scale=1 to keep the 6-hour lead time. The stats and invariants
are copied unchanged (same grid and channels).

    python scripts/subset_era5.py --src /path/latlon_025deg_hdf5_1h \\
        --dst /path/latlon_025deg_hdf5_6h_subset --years 2016:30 2017:5
"""

import argparse
import os
import shutil
import time

import h5py


def copy_year(src, dst, n_days, stride):
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        n = min(n_days * 24, fin["time"].shape[0])
        idx = range(0, n, stride)
        for key in ("channel", "latitude", "longitude"):
            fout.create_dataset(key, data=fin[key][:])
        fout.create_dataset("time", data=fin["time"][0:n:stride])
        data_in = fin["data"]
        data_out = fout.create_dataset("data", shape=(len(idx), *data_in.shape[1:]), dtype=data_in.dtype)
        for i, t in enumerate(idx):
            data_out[i] = data_in[t]
    return len(idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--years", nargs="+", required=True, help="year:days, e.g. 2016:30")
    parser.add_argument("--stride", type=int, default=6, help="keep every stride-th hour")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for sub in ("stats", "invar"):
        shutil.copytree(os.path.join(args.src, sub), os.path.join(args.dst, sub), dirs_exist_ok=True)
    for spec in args.years:
        year, days = spec.split(":")
        start = time.time()
        n = copy_year(os.path.join(args.src, f"{year}.h5"), os.path.join(args.dst, f"{year}.h5"), int(days), args.stride)
        print(f"{year}: {n} snapshots in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
