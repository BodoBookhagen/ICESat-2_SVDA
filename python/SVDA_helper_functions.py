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

def to_gdf(df):
    """Function to convert dataframe to geodataframe and reproject"""
    df = df.drop(columns=['geometry'])
    df['geometry'] = df.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
    df = gp.GeoDataFrame(df)
    df = df.drop(columns=['Latitude','Longitude'])
    df.set_crs(epsg=4326, inplace=True)
    df =df.to_crs(epsg=32733)
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



def getCoordRotFwd(xIn,yIn,R_mat,xRotPt,yRotPt,desiredAngle):
    """ The functions below are used to calculate the along-track distance, the functions are the same used by
    PhoREAL (Photon Research and Engineering Analysis Library) https://github.com/icesat-2UT/PhoREAL"""

    # Get shape of input X,Y data
    xInShape = np.shape(xIn)
    yInShape = np.shape(yIn)

    # If shape of arrays are (N,1), then make them (N,)
    xIn = xIn.ravel()
    yIn = yIn.ravel()

    # Suppress warnings that may come from np.polyfit
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # endif

    # If Rmatrix, xRotPt, and yRotPt are empty, then compute them
    if(len(R_mat)==0 and len(xRotPt)==0 and len(yRotPt)==0):

        # Get current angle of linear fit data
        x1 = xIn[0]
        x2 = xIn[-1]
        y1 = yIn[0]
        y2 = yIn[-1]
        # endif
        deltaX = x2 - x1
        deltaY = y2 - y1
        theta = np.arctan2(deltaY,deltaX)

        # Get angle to rotate through
        phi = np.radians(desiredAngle) - theta

        # Get rotation matrix
        R_mat = np.matrix(np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]]))

        # Get X,Y rotation points
        xRotPt = x1
        yRotPt = y1

    else:

        # Get angle to rotate through
        phi = np.arccos(R_mat[0,0])

    # endif

    # Translate data to X,Y rotation point
    xTranslated = xIn - xRotPt
    yTranslated = yIn - yRotPt

    # Convert np array to np matrix
    xTranslated_mat = np.matrix(xTranslated)
    yTranslated_mat = np.matrix(yTranslated)

    # Get shape of np X,Y matrices
    (xTranslated_matRows,xTranslated_matCols) = xTranslated_mat.shape
    (yTranslated_matRows,yTranslated_matCols) = yTranslated_mat.shape

    # Make X input a row vector
    if(xTranslated_matRows > 1):
        xTranslated_mat = np.transpose(xTranslated_mat)
    #endif

    # Make Y input a row vector
    if(yTranslated_matRows > 1):
        yTranslated_mat = np.transpose(yTranslated_mat)
    #endif

    # Put X,Y data into separate rows of matrix
    xyTranslated_mat = np.concatenate((xTranslated_mat,yTranslated_mat))

    # Compute matrix multiplication to get rotated frame
    measRot_mat = np.matmul(R_mat,xyTranslated_mat)

    # Pull out X,Y rotated data
    xRot_mat = measRot_mat[0,:]
    yRot_mat = measRot_mat[1,:]

    # Convert X,Y matrices back to np arrays for output
    xRot = np.array(xRot_mat)
    yRot = np.array(yRot_mat)

    # Make X,Y rotated output the same shape as X,Y input
    xRot = np.reshape(xRot,xInShape)
    yRot = np.reshape(yRot,yInShape)

    # Reset warnings
    warnings.resetwarnings()

    # Return outputs
    return xRot, yRot, R_mat, xRotPt, yRotPt, phi

class AtlRotationStruct:

    # Define class with designated fields
    def __init__(self, R_mat, xRotPt, yRotPt, desiredAngle, phi):

        self.R_mat = R_mat
        self.xRotPt = xRotPt
        self.yRotPt = yRotPt
        self.desiredAngle = desiredAngle
        self.phi = phi

def get_atl_alongtrack(df):
    """Function to calculate the along-track distance"""
    easting = np.array(df['Easting'])
    northing = np.array(df['Northing'])

    desiredAngle = 90
    crossTrack, alongTrack, R_mat, xRotPt, yRotPt, phi = \
    getCoordRotFwd(easting, northing, [], [], [], desiredAngle)

    df = pd.concat([df,pd.DataFrame(crossTrack, columns=['crosstrack'])],axis=1)
    df = pd.concat([df,pd.DataFrame(alongTrack, columns=['alongtrack'])],axis=1)

    rotation_data = AtlRotationStruct(R_mat, xRotPt, yRotPt, desiredAngle, phi)

    return df, rotation_data
