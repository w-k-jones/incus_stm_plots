#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import argparse
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import satpy
from satpy import Scene
from pyresample.geometry import AreaDefinition
import pyproj
pyproj.datadir.set_data_dir("/home/users/wkjones/miniforge3/envs/tobac_flow/share/proj")

parser = argparse.ArgumentParser()
parser.add_argument("file", help="SEVIRI file to regrid", type=str)
args = parser.parse_args()

file = pathlib.Path(args.file)

save_dir = pathlib.Path(r"/work/scratch-nopw2/wkjones/seviri_regrid")

if not file.exists():
    raise ValueError(f'File {args.file} does not exist')

EPSG_4326_900x900 = AreaDefinition(
    "EPSG_4326_900x900", 
    "Global equal latitude/longitude grid at 0.2 degree resolution", 
    "longlat",
    projection=4326,
    width=900, height=900, area_extent=(-90,-90,90,90)
)

def load_regrid_seviri(filename):
    scn = Scene([filename], reader="seviri_l1b_native")
    scn.load(["IR_087", "IR_108", "IR_120", "WV_062", "WV_073"])
    scn = scn.resample(EPSG_4326_900x900)
    grid_ds = scn.to_xarray()
    grid_ds = grid_ds.rename_vars(IR_108="bt")
    grid_ds["wvd"] = grid_ds.WV_062 - grid_ds.WV_073
    grid_ds["swd"] = grid_ds.IR_087 - grid_ds.IR_120
    grid_ds = grid_ds.drop_vars(["IR_087", "IR_120", "WV_062", "WV_073"])
    return grid_ds

ds = load_regrid_seviri(file).compute()

comp = dict(zstd=True, complevel=5, shuffle=True)
for var in ds.data_vars:
    ds[var].encoding.update(comp)

ds.to_netcdf(save_dir / f'{file.stem}.nc')