#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import warnings
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("frame", help="Frame to plot", type=int)
args = parser.parse_args()

frame = args.frame - 1

seviri_ds = xr.open_dataset("seviri_tracking.nc").isel(t=frame).load()
synsat_ds = xr.open_dataset("synsat_tracking.nc").isel(t=frame).load()

fig, axes = plt.subplots(2, 1, subplot_kw=dict(projection=ccrs.PlateCarree()), dpi=150, figsize=(8,6.4))
axes[0].set_extent([-80,80,-30,30], crs=ccrs.PlateCarree())
img = axes[0].imshow(seviri_ds.bt.values, cmap="binary", vmax=320, vmin=180, extent=[-90,90,-90,90], origin="upper")
# axes[0].imshow(seviri_ds.swd.values, cmap="RdBu", vmax=10, vmin=-10, extent=[-90,90,-90,90], origin="upper")
cntr = [
    axes[0].contour(seviri_ds.longitude, seviri_ds.latitude, seviri_ds.thin_anvil_labels>0, [0.5], colors=["C00"], transform_first=True), 
    axes[0].contour(seviri_ds.longitude, seviri_ds.latitude, seviri_ds.thick_anvil_labels>0, [0.5], colors=["C01"], transform_first=True), 
    axes[0].contour(seviri_ds.longitude, seviri_ds.latitude, seviri_ds.core_labels>0, [0.5], colors=["C03"], transform_first=True), 
]
axes[0].coastlines()
axes[0].set_title(f'SEVIRI', loc="left")
axes[0].set_title(f'{np.datetime64(seviri_ds.t.item(), "ns").astype("datetime64[s]")}', loc="right")

axes[1].set_extent([-80,80,-30,30], crs=ccrs.PlateCarree())
img = axes[1].imshow(synsat_ds.bt.values, cmap="binary", vmax=320, vmin=180, extent=[-90,90,-90,90], origin="lower")
# axes[1].imshow(synsat_ds.wvd.values, cmap="RdBu", vmax=0, vmin=-15, extent=[-90,90,-90,90], origin="lower")
cntr = [
    axes[1].contour(seviri_ds.longitude, seviri_ds.latitude[::-1], synsat_ds.thin_anvil_labels>0, [0.5], colors=["C00"], transform_first=True), 
    axes[1].contour(seviri_ds.longitude, seviri_ds.latitude[::-1], synsat_ds.thick_anvil_labels>0, [0.5], colors=["C01"], transform_first=True), 
    axes[1].contour(seviri_ds.longitude, seviri_ds.latitude[::-1], synsat_ds.core_labels>0, [0.5], colors=["C03"], transform_first=True), 
]
axes[1].coastlines()
axes[1].set_title(f'ICON NGC-4008a SynSat', loc="left")
axes[1].set_title(f'{np.datetime64(synsat_ds.t.item(), "ns").astype("datetime64[s]")}', loc="right")


ax_position = axes[1].get_position().bounds
ax_corner = ax_position[0]
ax_width = ax_position[2]

cax = fig.add_axes([ax_corner+ax_width*0.025, 0.05, ax_width*0.425, 0.025])
cbar = plt.colorbar(img, cax=cax, label=r"$10.8\,\mu m$ BT [K]", orientation="horizontal")

import matplotlib.lines as mlines

fig.legend(
    [mlines.Line2D([], [], color='C00'), mlines.Line2D([], [], color='C01'), mlines.Line2D([], [], color='C03')], 
    ['Thin anvil', 'Thick anvil', 'Growing core'], title="Tracked features:", 
    loc='center', bbox_to_anchor=(ax_corner + ax_width*0.5, -0.018, ax_width*0.5, 0.1), 
    ncol=2
)

save_dir = pathlib.Path("./seviri_synsat_frames")
fig.savefig(save_dir/f'seviri_synsat_{frame:02d}.png', bbox_inches="tight", pad_inches=0.2)