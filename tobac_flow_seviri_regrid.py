#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import warnings
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tobac_flow.flow import create_flow
from tobac_flow.detection import (
    detect_cores,
    get_anvil_markers,
    detect_anvils,
    relabel_anvils,
)

seviri_path = pathlib.Path(r"/work/scratch-nopw2/wkjones/seviri_regrid")

regrid_files = sorted(list(seviri_path.glob("*.nc")))

regrid_ds = xr.open_mfdataset(regrid_files, combine="nested", concat_dim="t").load()

regrid_ds = regrid_ds.assign_coords(t=pd.DatetimeIndex(
    pd.date_range(datetime(2021,7,1,4), datetime(2021,7,2,4), freq="900s", inclusive="left")
))

bt = regrid_ds.bt.compute()
wvd = regrid_ds.wvd.compute()
swd = regrid_ds.swd.compute()

flow = create_flow(
    bt, model="Farneback", vr_steps=1, smoothing_passes=1, interp_method="linear"
)

wvd_threshold = 0.25
bt_threshold = 0.25
overlap = 0.5
absolute_overlap = 1
subsegment_shrink = 0.0
min_length = 2

core_labels = detect_cores(
    flow,
    bt,
    wvd,
    swd,
    wvd_threshold=wvd_threshold,
    bt_threshold=bt_threshold,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    subsegment_shrink=subsegment_shrink,
    min_length=min_length,
    use_wvd=False,
)

upper_threshold = -5
lower_threshold = -12.5
erode_distance = 2

anvil_markers = get_anvil_markers(
    flow,
    wvd - np.maximum(swd, 0),
    threshold=upper_threshold,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    subsegment_shrink=subsegment_shrink,
    min_length=min_length,
)

print("Final thick anvil markers: area =", np.sum(anvil_markers != 0), flush=True)
print("Final thick anvil markers: n =", anvil_markers.max(), flush=True)

thick_anvil_labels = detect_anvils(
    flow,
    wvd - np.maximum(swd, 0),
    markers=anvil_markers,
    upper_threshold=upper_threshold,
    lower_threshold=lower_threshold,
    erode_distance=erode_distance,
    min_length=min_length,
)
print("Initial detected thick anvils: area =", np.sum(thick_anvil_labels != 0), flush=True)
print("Initial detected thick anvils: n =", thick_anvil_labels.max(), flush=True)

thin_anvil_labels = detect_anvils(
    flow,
    wvd + swd,
    markers=thick_anvil_labels,
    upper_threshold=upper_threshold,
    lower_threshold=lower_threshold,
    erode_distance=erode_distance,
    min_length=min_length,
)

print("Detected thin anvils: area =", np.sum(thin_anvil_labels != 0), flush=True)
print("Detected thin anvils: n =", np.max(thin_anvil_labels), flush=True)

regrid_ds["core_labels"] = bt.copy(data=core_labels)
regrid_ds["thick_anvil_labels"] = bt.copy(data=thick_anvil_labels)
regrid_ds["thin_anvil_labels"] = bt.copy(data=thin_anvil_labels)

comp = dict(zstd=True, complevel=5, shuffle=True)
for var in regrid_ds.data_vars:
    regrid_ds[var].encoding.update(comp)

regrid_ds.to_netcdf("seviri_tracking.nc")

