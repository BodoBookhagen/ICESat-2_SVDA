# ICESat-2_SVDA

Python codes and Jupyter Notebooks for the "Sparse Vegetation Detection Algorithm" (SVDA) to process ICESat-2 ATL03 data.

The concept and theoretical background is described in the journal article "Measuring vegetation height and their seasonal changes in western Namibia using spaceborne lidars (GEDI and ICESat-2)"

## Jupyter Notebooks
In order to run the codes and follow the steps in the Jupyter Notebooks, you will need to have installed several additional packages. This can be done through [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

Here is a code-snippet to install the required packages into an conda environment called `icesat2`:

```
conda create -y -n icesat2 -c conda-forge ipython numpy python \
  ipython matplotlib h5py pandas scipy pyproj pip fiona shapely \
  jupyter ipywidgets gdal tqdm scikit-learn weightedstats \
  geopandas cartopy plotly
conda activate icesat2
pip install tables laspy requests
```

You can also add this conda environment to be recognized by the Jupyter Notebooks:
```
python -m ipykernel install --user --name=icesat2
```

In addition, you may want to install the python tools for obtaining and working with elevation data from the NASA ICESat-2 mission from github. This relies on MPI/OPENMPI and you may need to make sure to install the correct version of the compilers within conda. Alternatively, you can install the packages to the system. The following snippet works on an Ubuntu 18.04 and 20.04, but should be easily transferable to other systems.


```
cd ~
conda activate icesat2
pip install pyGEDI
conda install -y mpi4py
conda install -c -y conda-forge openmpi-mpicc
conda install -c -y conda-forge c-compiler compilers cxx-compiler
git clone https://github.com/tsutterley/read-ICESat-2.git
cd read-ICESat-2
export OMPI_MCA_opal_cuda_support=true
python setup.py build
python setup.py install
cd ~
```

Now you should be ready to run
