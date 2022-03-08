import pandas as pd
import geopandas as gp
import numpy as np
import datetime, sys, warnings
from pyproj import Transformer


def jdtodatestd (jdate):
    """Convert Julian date to calendar date"""
    fmt = '%Y%j%'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)

def to_gdf(df, dst_crs=32733):
    """Function to convert dataframe to geodataframe and reproject"""
    df = df.drop(columns=['geometry'])
    df['geometry'] = df.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
    df = gp.GeoDataFrame(df)
    df = df.drop(columns=['Latitude','Longitude'])
    df.set_crs(epsg=4326, inplace=True)
    df =df.to_crs(epsg=dst_crs)
    return df


# Functions to calculate the histograms
def canop_hist(df):
    """Canopy height and elevation diffrences distributions"""
    elev_bins = np.arange(-30, 30, 0.25)
    bins, bins_edges = np.histogram(df, elev_bins)
    bins_centers = bins_edges[:-1] + np.diff(bins_edges)/2
    mean = round(np.average(bins_centers, weights=bins),2)
    var = np.average((bins_centers - mean)**2, weights=bins)
    std = round(np.sqrt(var),2)
    med = round(ws.numpy_weighted_median(bins_centers, weights= bins),2)

    return bins_centers, bins, mean, std, med

def elev_hist(df):
    """Ground elevation distributions"""
    elev_bins = np.arange(1000, 1500, 10)
    bins, bins_edges = np.histogram(df, elev_bins)
    bins_centers = bins_edges[:-1] + np.diff(bins_edges)/2
    mean = round(np.average(bins_centers, weights=bins),2)
    var = np.average((bins_centers - mean)**2, weights=bins)
    std = round(np.sqrt(var),2)
    med = round(ws.numpy_weighted_median(bins_centers, weights= bins),2)

    return bins_centers, bins, mean, std, med

def KDE_plot(ddf, wdf):
    """Function to calculate Kernel Density Estimination"""

    x1 = ddf['S2_NDVI_Diff_2020']
    y1 = ddf['Canopy_Height']

    x2 = wdf['S2_NDVI_Diff_2020']
    y2 = wdf['Canopy_Height']

    xmin, xmax = min(x2), max(x2)
    ymin, ymax = min(y2)-0.25, max(y2)+0.25

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values1 = np.vstack([x1, y1])
    kernel1 = st.gaussian_kde(values1)
    f1 = np.reshape(kernel1(positions).T, xx.shape)

    #Calculate and scale densities
    values2 = np.vstack([x2, y2])
    kernel2 = st.gaussian_kde(values2)
    f2 = np.reshape(kernel2(positions).T, xx.shape)
    ff1 = f1/np.max(f1)
    ff2 = f2/np.max(f2)

    return x1,x2, y1,y2, ff1, ff2

def GetExtent(ds):
    """Read Geotiff to numpy array and get the extend of the geotiff"""
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return (xmin, xmax, ymin, ymax)
