import os, h5py, glob, sys, warnings, tqdm
import pandas as pd
import numpy as np
import geopandas as gp
from pyproj import Transformer
from pyproj import proj
from operator import attrgetter
from scipy import stats
from scipy.interpolate import interp1d
from scipy import signal
from math import tan, pi
import scipy.spatial as spatial
#from sklearn.preprocessing import StandardScaler

def ATL03_signal_photons(fname, ATL03_output_path, ROI_fname, EPSG_Code, reprocess=False):
    """
    ATL03_signal_photons(fname, ATL03_output_path, ROI_fname, EPSG_Code)

    Takes a ATL03 H5 file, extracts the following attributes for
    each beam (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r):

    heights/lat_ph
    heights/lon_ph
    heights/h_ph
    heights/dist_ph_along
    heights/signal_conf_ph

    The function extracts along-track distances,
    converts latitude and longitude to local UTM coordinates,
    filters out land values within the geographic area <ROI_fname>,
    usually a shapefile in EPSG:4326 coordinates and writes these
    to a compressed HDF file in <ATL03_output_path> starting with 'Land_'
    and the date and time of the beam.

    """
    ATL03 = h5py.File(fname,'r')

    gtr = [g for g in ATL03.keys() if g.startswith('gt')]

    ATL03_objs = []
    ATL03.visit(ATL03_objs.append)
    ATL03_SDS = [o for o in ATL03_objs if isinstance(ATL03[o], h5py.Dataset)]

    # Retrieve datasets
    for b in gtr:
        print('Opening %s: %s'%(os.path.basename(fname), b))
        if reprocess==False and os.path.exists(os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b))):
            print('File %s already exists and reprocessing is turned off.'%os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)))
            continue
        attribute_lat_ph = b + '/heights/lat_ph'
        lat_ph = np.asarray(ATL03[attribute_lat_ph]).tolist()
        attribute_lon_ph = b + '/heights/lon_ph'
        lon_ph = np.asarray(ATL03[attribute_lon_ph]).tolist()
        attribute_h_ph = b + '/heights/h_ph'
        h_ph = np.asarray(ATL03[attribute_h_ph]).tolist()
        attribute_dist_ph_along = b + '/heights/dist_ph_along'
        dist_ph_along = np.asarray(ATL03[attribute_dist_ph_along]).tolist()
        attribute_signal_conf_ph = b + '/heights/signal_conf_ph'
        signal_conf_ph = np.asarray(ATL03[attribute_signal_conf_ph]).tolist()

#         lat_ph, lon_ph, h_ph, dist_ph_along, signal_conf_ph, gtx = ([] for i in range(6))
#         [lat_ph.append(h) for h in ATL03[[g for g in ATL03_SDS if g.endswith('/lat_ph') and b in g][0]][()]]
#         [lon_ph.append(h) for h in ATL03[[g for g in ATL03_SDS if g.endswith('/lon_ph') and b in g][0]][()]]
#         [h_ph.append(h) for h in ATL03[[g for g in ATL03_SDS if g.endswith('/h_ph') and b in g][0]][()]]
#         [dist_ph_along.append(h) for h in ATL03[[g for g in ATL03_SDS if g.endswith('/dist_ph_along') and b in g][0]][()]]
#         [signal_conf_ph.append(h) for h in ATL03[[g for g in ATL03_SDS if g.endswith('/signal_conf_ph') and b in g][0]][()]]

        ATL03_df = pd.DataFrame({'Latitude': lat_ph, 'Longitude': lon_ph, 'Along-track_Distance': dist_ph_along,
                                 'Photon_Height': h_ph, 'Signal_Confidence':signal_conf_ph})

        del lat_ph, lon_ph, h_ph, dist_ph_along, signal_conf_ph
        del attribute_lat_ph, attribute_lon_ph, attribute_h_ph, attribute_dist_ph_along, attribute_signal_conf_ph

        ATL03_df.loc[:, 'Land'] = ATL03_df.Signal_Confidence.map(lambda x: x[0])
        ATL03_df = ATL03_df.drop(columns=['Signal_Confidence'])

        # Transform coordinates into utm
        x, y = np.array(ATL03_df['Longitude']), np.array(ATL03_df['Latitude'])
        transformer = Transformer.from_crs('epsg:4326', EPSG_Code, always_xy=True)
        xx, yy = transformer.transform(x, y)

        # Save the utm coordinates into the dataframe
        ATL03_df['Easting'] = xx
        ATL03_df['Northing'] = yy

        ATL03_df, rotation_data = get_atl_alongtrack(ATL03_df)

        ROI = gp.GeoDataFrame.from_file(ROI_fname, crs='EPSG:4326')
        minLon, minLat, maxLon, maxLat = ROI.envelope[0].bounds

        # Subset the dataframe into the study area bounds
        ATL03_df = ATL03_df.where(ATL03_df['Latitude'] > minLat)
        ATL03_df = ATL03_df.where(ATL03_df['Latitude'] < maxLat)
        ATL03_df = ATL03_df.where(ATL03_df['Longitude'] > minLon)
        ATL03_df = ATL03_df.where(ATL03_df['Longitude'] < maxLon)
        ATL03_df = ATL03_df.dropna()

        if not os.path.exists(ATL03_output_path):
            os.mkdir(ATL03_output_path)

        #ATL03_df.to_csv('{}_{}.csv'.format(ATL03_files[23:-3], gtr), header=True)
        ATL03_df.to_hdf(os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)),
                        key='%s_%s'%('_'.join(os.path.basename(fname).split('_')[1::])[:-4],b), complevel=7)
        print('saved to %s'%os.path.join(ATL03_output_path,'ATL03_Land_%s_%s.hdf'%('_'.join(os.path.basename(fname).split('_')[1:2]),b)))
        print()
    ATL03.close()

def ATL03_ground_preliminary_canopy_photons(fname, ATL03_output_path, reprocess=False):
    """
    ATL03_ground_preliminary_canopy_photons(fname, ATL03_output_path, reprocess=False)

    Takes a ATL03_Land_* file generated with ATL03_signal_photons.

    Bins data into 30-m along track bins and uses
    topographic information to detrend data. Next, takes photons
    between the 25th and 75th photon height percentiles to identify ground.
    Preliminary canopy is above the 75th percentile.

    """

    print('Opening %s: '%os.path.basename(fname), end='')
    if reprocess==False and os.path.exists(os.path.join(ATL03_output_path,'ATL03_PreCanopy_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])) and os.path.exists(os.path.join(ATL03_output_path,'ATL03_Ground_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])):
        print('Files %s and %s already exist and reprocessing is turned off.'%(os.path.join(ATL03_output_path,'ATL03_PreCanopy_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]),
        os.path.join(ATL03_output_path,'ATL03_Ground_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])))
        return
    ATL03_df = pd.read_hdf(fname, mode='r')

    # Bining in the along-track direction (bins of 30 m)
    rows_labels = [f"[{i}, {i+30}])" for i in range(int(min(ATL03_df['alongtrack'])), int(max(ATL03_df['alongtrack'])), 30)]
    rows_bins = pd.IntervalIndex.from_tuples([(i, i+30) for i in range(int(min(ATL03_df['alongtrack'])), int(max(ATL03_df['alongtrack'])), 30)], closed="left")
    rows_binned = pd.cut(ATL03_df['alongtrack'], rows_bins, labels=rows_labels, precision=2, include_lowest=True)
    rows_binned.sort_values(ascending=True, inplace=True)

    rows_binned.drop_duplicates(keep='first',inplace=True)
    rows_left = np.asarray(rows_binned.map(attrgetter('left')))
    rows_right = np.asarray(rows_binned.map(attrgetter('right')))
    rows_left = rows_left[~np.isnan(rows_left)]
    rows_right = rows_right[~np.isnan(rows_right)]

    ground_ph = []
    canop = []
    print('Detrending topography and photon-height filtering... ')
    for i in tqdm.tqdm(range(len(rows_left))):
        df = ATL03_df.where((ATL03_df['alongtrack']>= rows_left[i]) & (ATL03_df['alongtrack'] < rows_right[i]))
        df = df.dropna()

        # Topography detrending and outliers filtering
        df['detrend'] = signal.detrend(df['Photon_Height'])
        df = df.where((df['detrend']> -(30*tan((pi/180)*30))) & (df['detrend'] < (30*tan((pi/180)*30))))
        df = df.dropna()
        df = df.drop(columns=['detrend'])

        # Retrieving photons between the 25th and the 75th percentiles
        if len(df)>5:
            df_grd = df.where((df['Photon_Height'] <= np.percentile(df['Photon_Height'],75)) & (df['Photon_Height'] > np.percentile(df['Photon_Height'],25)))
            df_grd = df_grd.dropna()

            #Topography detrend and outliers filtering
            df_grd['detrend'] = signal.detrend(df_grd['Photon_Height'])
            df_grd = df_grd.where((df_grd['detrend']> -(tan((pi/180)*30))) & (df_grd['detrend'] < (tan((pi/180)*30))))
            df_grd = df_grd.dropna()

        # Photons with height above the maximum of each 25th-75th bin are retrieved as preliminary canopy photons
            if len(df_grd)>5:
                df_canop = df.where(df['Photon_Height'] > max(df_grd['Photon_Height']))
                df_canop = df_canop.dropna()
                canop.append(df_canop)
                ground_ph.append(df_grd)

    # Preliminary canopy photons
    canop = [j for j in canop if len(j)>=0]
    if len(canop)>0:
        Canopy = pd.concat(canop, axis=0)

    latitude, longitude, along,cross, med, north, east = ([] for i in range(7))

    ground_ph = [j for j in ground_ph if len(j)!=0]

    # Final ground photons are the medians of each bin placed in the center of each bin
    for df in ground_ph:
        al = (max(df['alongtrack']) + min(df['alongtrack']))/2
        along.append(al)
        es = (max(df['Easting']) + min(df['Easting']))/2
        east.append(es)
        nd = (max(df['Northing']) + min(df['Northing']))/2
        north.append(nd)
        cr = (max(df['crosstrack']) + min(df['crosstrack']))/2
        cross.append(cr)
        lat = (max(df['Latitude']) + min(df['Latitude']))/2
        latitude.append(lat)
        lon = (max(df['Longitude']) + min(df['Longitude']))/2
        longitude.append(lon)
        m = np.median(df['Photon_Height'])
        med.append(m)

    # Ground photons dataframe
    median_df = pd.DataFrame({'Latitude': latitude, 'Longitude': longitude, 'alongtrack': along, 'crosstrack': cross,
                              'Easting': east, 'Northing': north, 'Photon_Height':med})

    # Saving the final ground photons into hdf file
    if len(median_df) >5:
        median_df.to_hdf(os.path.join(ATL03_output_path,'ATL03_Ground_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]),
                         key='Ground_%s'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])
        print("saved to %s"%os.path.join(ATL03_output_path,'ATL03_Ground_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]) )

        # calculate the canopy height by subtracting the ground height of each canopy photon by interpolating the final ground photons
        if len(Canopy) > 5:
            Canopy = Canopy.where((Canopy['alongtrack'] > min(median_df['alongtrack'])) & (Canopy['alongtrack'] < max(median_df['alongtrack'])))
            Canopy = Canopy.dropna()
            f_interp1d = interp1d(median_df['alongtrack'], median_df['Photon_Height'], kind='cubic')
            Canopy['Canopy_Height'] = Canopy['Photon_Height'] - f_interp1d(Canopy['alongtrack'])

            Canopy.to_hdf(os.path.join(ATL03_output_path,'ATL03_PreCanopy_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]),
                          key='PreCanopy_%s'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])
            print("saved to %s"%os.path.join(ATL03_output_path,'ATL03_PreCanopy_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]) )
    print()


def ATL03_canopy_and_top_of_canopy_photons(fname, ATL03_output_path, reprocess=False):
    """
    ATL03_canopy_and_top_of_canopy_photons(fname, ATL03_output_path, reprocess)

    Takes a ATL03_PreCanopy_* HDF file generated with ATL03_ground_preliminary_canopy_photons.

    Saves Canopy and Top-Of-Canopy (TOC) to a new hdf file.

    """

    print('Opening %s: '%os.path.basename(fname),end='')
    if reprocess==False and os.path.exists(os.path.join(ATL03_output_path,'ATL03_TOC_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])):
        print('File %s already exists and reprocessing is turned off.'%(os.path.join(ATL03_output_path,'ATL03_TOC_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])))
        return
    Canopy = pd.read_hdf(fname, mode='r')

    Canopy = Canopy.where(Canopy['Canopy_Height']>=3)
    Canopy = Canopy.dropna()

    # Easting, Northing and Canopy Height scaling
    if len(Canopy)>5:
        df = Canopy
        df['z'] = (df['Canopy_Height']-np.min(df['Canopy_Height']))/(np.max(df['Canopy_Height'])-np.min(df['Canopy_Height']))
        df['x'] = (df['alongtrack']-np.min(df['alongtrack']))/(np.max(df['alongtrack'])-np.min(df['alongtrack']))

        X = np.array(df[['x', 'z']])
        T = spatial.cKDTree(X)

        index = []
        for i in range(len(df)):
            idx = T.query_ball_point(X[i], r = 0.01)
            index.append(len(idx))
        df['NN'] = index

        # Filtering out canopy photons with number of neighbors below the 15th percentile of the number of neighbors
        ddf = df.where(df['NN'] >=6)
        ddf = ddf.dropna()

        # Binning the along-track direction (bins of 10 m)
        if len(ddf)>0:
            rows_labels = [f"[{i}, {i+10}])" for i in range(int(min(ddf['alongtrack'])), int(max(ddf['alongtrack'])), 10)]
            rows_bins = pd.IntervalIndex.from_tuples([(i, i+10) for i in range(int(min(ddf['alongtrack'])),
                                                                               int(max(ddf['alongtrack'])), 10)], closed="left")

            rows_binned = pd.cut(ddf['alongtrack'], rows_bins, labels=rows_labels, precision=2, include_lowest=True)
            rows_binned.sort_values(ascending=True, inplace=True)

            rows_binned.drop_duplicates(keep='first',inplace=True)
            rows_left = np.asarray(rows_binned.map(attrgetter('left')))
            rows_right = np.asarray(rows_binned.map(attrgetter('right')))
            rows_left = rows_left[~np.isnan(rows_left)]
            rows_right = rows_right[~np.isnan(rows_right)]

            df_canop = []
            print('Filtering along track...')
            for i in tqdm.tqdm(range(len(rows_left))):
                dff = ddf.where((ddf['alongtrack']>= rows_left[i]) & (ddf['alongtrack'] < rows_right[i]))
                dff = dff.dropna()
                dff['Photons_Numb'] = len(dff)
            # Calculate the maximum photon height of each bin
                if len(dff)>0:
                    dd = dff.where(dff['Canopy_Height']==max(dff['Canopy_Height']))
                    dd = dd.dropna()
                    df_canop.append(dd)
            df_canop = [j for j in df_canop if len(j)>=0]
            if len(df_canop)>0:
                toc = pd.concat(df_canop, axis=0)

                # Save the canopy and top of canopy photons into new hdf files
                toc.to_hdf(os.path.join(ATL03_output_path,'ATL03_TOC_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]),
                          key='TOC_%s'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])
                print('saved to %s'%os.path.join(ATL03_output_path,'ATL03_TOC_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]))
    print()

def ATL03_GrassHeight_photons(fname, ATL03_output_path, reprocess=False):
    """
    ATL03_GrassHeight_photons(fname, ATL03_output_path, reprocess)

    Takes a ATL03_PreCanopy_* HDF file generated with ATL03_ground_preliminary_canopy_photons.

    Saves grass heights into a new HDF.

    """
    print('Opening %s: '%os.path.basename(fname),end='')
    if reprocess==False and os.path.exists(os.path.join(ATL03_output_path,'ATL03_GrassHeight_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])):
        print('File %s already exists and reprocessing is turned off.'%(os.path.join(ATL03_output_path,'ATL03_GrassHeight_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])))
        return
    Canopy = pd.read_hdf(fname, mode='r')

    # All photons with canopy height between 0.5 m and 3 m
    Canopy = Canopy.where((Canopy['Canopy_Height']>=0.5) & (Canopy['Canopy_Height']<3))
    Canopy = Canopy.dropna()

    # Easting, Northing and Canopy Height scaling
    if len(Canopy)>5:
        df = Canopy
        df['z'] = (df['Canopy_Height']-np.min(df['Canopy_Height']))/(np.max(df['Canopy_Height'])-np.min(df['Canopy_Height']))
        df['x'] = (df['alongtrack']-np.min(df['alongtrack']))/(np.max(df['alongtrack'])-np.min(df['alongtrack']))

        X = np.array(df[['x', 'z']])
        T = spatial.cKDTree(X)

        index = []
        for i in range(len(df)):
            idx = T.query_ball_point(X[i], r = 0.01)
            index.append(len(idx))
        df['NN'] = index

        # Filtering out canopy photons with number of neighbors below the 15th percentile of the number of neighbors
        ddf = df.where(df['NN'] >=6)
        ddf = ddf.dropna()

        # Binning the along-track direction (bins of 10 m)
        if len(ddf)>0:
            rows_labels = [f"[{i}, {i+10}])" for i in range(int(min(ddf['alongtrack'])), int(max(ddf['alongtrack'])), 10)]
            rows_bins = pd.IntervalIndex.from_tuples([(i, i+10) for i in range(int(min(ddf['alongtrack'])),
                                                                               int(max(ddf['alongtrack'])), 10)], closed="left")

            rows_binned = pd.cut(ddf['alongtrack'], rows_bins, labels=rows_labels, precision=2, include_lowest=True)
            rows_binned.sort_values(ascending=True, inplace=True)

            rows_binned.drop_duplicates(keep='first',inplace=True)
            rows_left = np.asarray(rows_binned.map(attrgetter('left')))
            rows_right = np.asarray(rows_binned.map(attrgetter('right')))
            rows_left = rows_left[~np.isnan(rows_left)]
            rows_right = rows_right[~np.isnan(rows_right)]

            df_canop = []
            for i in range(len(rows_left)):
                dff = ddf.where((ddf['alongtrack']>= rows_left[i]) & (ddf['alongtrack'] < rows_right[i]))
                dff = dff.dropna()
            # Calculate the maximum photon height of each bin
                if len(dff)>0:
                    dd = dff.where(dff['Canopy_Height']==max(dff['Canopy_Height']))
                    dd = dd.dropna()
                    df_canop.append(dd)
            df_canop = [j for j in df_canop if len(j)>=0]
            if len(df_canop)>0:
                grassheight = pd.concat(df_canop, axis=0)
                # Save the grass height into new HDF files
                grassheight.to_hdf(os.path.join(ATL03_output_path,'ATL03_GrassHeight_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]),                          key='GrassHeight_%s'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4])
                print('saved to %s'%os.path.join(ATL03_output_path,'ATL03_GrassHeight_%s.hdf'%'_'.join(os.path.basename(fname).split('_')[2::])[:-4]))
        print()
