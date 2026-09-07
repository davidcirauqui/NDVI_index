# NDVI index

I assess the health of vegetation over a given geographic area using the Normalized Difference Vegetation Index (NDVI), computed from Sentinel-2 satellite imagery. To do so, the notebook is separated in two sections:

- **A first glance**, over Sant Antoni de Portmany (Balearic Islands): monthly NDVI snapshots over a year to build intuition for how water, urban areas, and cropfields show up in the index.
- **Desertification monitoring in the Sahel** (Ferlo desert, Senegal): a decade of monthly NDVI data (2016–present) analyzed for trend, deseasonalized, cross-correlated with rainfall, and broken down into phenology metrics (green-up, senescence, season length, peak timing).

The notebook finished with a conclusions section, in which I summarise what the gathered data can tell about the desertification within the studied area.


## Structure
- [NDVI.ipynb](NDVI.ipynb): the notebook going through the two case studies.
- [utils.py](utils.py): developed functions, imported in the notebook as 'NDVI'.


## Setup

You will first need to create a [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) account and register an OAuth client to get a `client_id` / `client_secret` pair in order to get access to Sentinel-2 data through SentinelHub. When runing the notebook, the configuration cell will promt for your `client_id` and `client_secret` (via `getpass`, so they are not sroted in the notebook), and save the SentinelHub profile locally.

Furthermore, the notebook requires Python 3.12 and the following packages:
- numpy
- pandas
- scipy
- requests
- matplotlib
- sentinelHub



## Data source
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) for NDVI 
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) for rainfall data 




