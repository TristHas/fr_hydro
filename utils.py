from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd

import xarray as xr 
import networkx as nx
import torch

import hvplot.pandas

import diffhydro as dh
import diffhydro.pipelines as dhp
import xtensor as xt

GRAPH_ROOT = Path('/data_prediction005/SYSTEM/prediction002/home/tristan/data/RIVER_GRAPH')

def load_graph_data(name, load_basins=False):
    root = GRAPH_ROOT / name
    catchments = gpd.GeoDataFrame(geometry=pd.read_pickle(root / "basins.pkl"))
    catchments = catchments.set_crs("epsg:4326")
    g = pd.read_pickle(root / "g.pkl" )

    df = pd.DataFrame(dict(g.nodes)).T
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    kp = pd.read_pickle(root / "kp.pkl")
    points["color"] = points["color"].fillna("black")
    if load_basins:
        basins = gpd.GeoDataFrame(geometry=pd.read_pickle(root / "catchments.pkl")).set_crs("epsg:4326")
        return g, points, catchments, kp, basins
    else:
        return g, points, catchments, kp

def _load_fr_data(root, kp): #inp_dyn_fp32_comp.nc
    inp_dyn = dh.io.open_dataset(root / "inp_dyn_fp32.nc")\
                .to_datatensor(dim="variable")\
                .rename({"polygon_index":"spatial"}).expand_dims("batch")\
                .transpose("batch", "spatial", "time","variable")
    
    inp_stat = dh.io.read_pickle(root / "inp_stat.pkl", dims=["spatial", "variable"])
    
    lbl = dh.io.read_pickle(root / "out.pkl", dims=["time", "spatial"])\
               .sel(spatial=kp["sta_code_h3"].values)\
               .assign_coords(spatial=kp.index)\
               .expand_dims("batch")\
               .transpose("batch", "spatial", "time")
    
    inp_time = inp_dyn["time"]
    lbl_time = lbl["time"]
    msk_time = inp_time[inp_time.isin(lbl_time)]
    
    lbl = lbl.sel(time=msk_time)
    inp_dyn = inp_dyn.sel(time=msk_time)
    return inp_stat, inp_dyn, lbl

def load_fr_data(root, device="cpu"):
    g, points, catchments, kp = load_graph_data("FR_10_base")
    g = select_subgraph(g)
    inp_stat, inp_dyn, lbl = _load_fr_data(root, kp)
    
    df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    df = df[['channel_length',"catchment_area","upa"]]
    df['channel_length']=df['channel_length'] * 90 / 1000
    df_param = (df - df.mean())  / df.std()
    
    g = dh.RivTree(g, "hayami", param_df=df_param, param_names=["upa"])

    ###
    ### Subgraph selection and ordering
    ###
    # static
    df = df.loc[g.nodes]
    
    # inputs
    inp_stat = inp_stat.sel(spatial=g.nodes)
    inp_stat = inp_stat.transpose("spatial", "variable")
    inp_stat = inp_stat.to(device=device, dtype=torch.float32)
    inp_stat = (inp_stat - inp_stat.mean("spatial")) / inp_stat.std("spatial")
    inp_stat = torch.nan_to_num(inp_stat, 0)
    
    inp_dyn=inp_dyn.sel(spatial=g.nodes).to(dtype=torch.float32)
    
    # lbl
    lbl_mask = lbl["spatial"].to_pandas()[lbl["spatial"].to_pandas().isin(g.nodes)].values
    lbl = lbl.sel(spatial=lbl_mask)
    lbl = lbl.to(dtype=torch.float32)
    
    ### Selecting additional variables
    cat_area = torch.from_numpy(df["catchment_area"].values).float().to(device)
    channel_dist = torch.from_numpy(df["channel_length"].values).float().to(device)
    basin_area = torch.from_numpy(df["upa"].values).float().to(device)
    statics = {"x_stat":inp_stat, "cat_area":cat_area, "channel_dist":channel_dist}
    return g, kp, inp_stat, inp_dyn, lbl, statics

def select_subgraph(g, outlet=31857435):
    return g.subgraph(nx.ancestors(g, outlet) | {outlet})

def split_and_normalize_data(inp, lbl):
    inp_tr = inp.sel(time=slice(None, "1989"))
    inp_te = inp.sel(time=slice("1990", None))
    
    inp_mean = 0#inp_tr.mean(dim=("time", "spatial"))
    inp_std  = inp_tr.std(dim=("time", "spatial"))
    
    inp_tr = (inp_tr - inp_mean) / inp_std
    inp_te = (inp_te - inp_mean) / inp_std
    
    lbl_tr  = lbl.sel(time=slice(None, "1989"))
    lbl_te  = lbl.sel(time=slice("1990", None))
    
    lbl_std = lbl_tr.mean()
    
    lbl_tr = lbl_tr / lbl_std
    lbl_te = lbl_te / lbl_std
    return inp_tr, inp_te, lbl_tr, lbl_te, lbl_std