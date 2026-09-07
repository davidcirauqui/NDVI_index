
import numpy as np
import pandas as pd
from scipy import stats
import requests

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    SentinelHubCatalog,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    MosaickingOrder,
)


def get_ndvi_mat(coords, resolution, dates, config):
    """Function that returns the NDVI index
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the NDVI index
        - resolution: Resolution of the requested data
        - dates: Dates delimiting the time span over which the data is requested. The function computes the NDVI index on the less cloudy day within that period of time
        - config: SentinelHub configuration
    outputs:
        - ndvi_mat: array containing the computed NDVI index over the specified coordinate box and dates
    """


    # SentinelHub specifications
    m_bbox = BBox(bbox = coords, crs=CRS.WGS84)
    m_size = bbox_to_dimensions(m_bbox, resolution = resolution)
    catalog = SentinelHubCatalog(config = config)

    # select NIR (B08) and RED (B04) channels from sentinel data
    evalscript_ndvi = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B04","B08"],
                    units: "DN"
                }],
                output: {
                    bands: 2,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B08];
        }
    """

    # define requesting function
    request_ndvi = SentinelHubRequest(
        evalscript = evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=(dates[0], dates[1]),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox = m_bbox,
        size = m_size,
        config = config,
    )

    # request data from SentinelHub
    data = request_ndvi.get_data()

    # process NIR and RED chanels' data into NDVI index
    n = data[0].reshape(data[0].shape[0] * data[0].shape[1], 2)
    df = pd.DataFrame(n)
    df.columns = ['red', 'nir']
    df['ndvi'] = (df.nir - df.red) / (df.nir + df.red)
    # df.head()

    # reshape into matrix for plotting purposes
    ndvi_mat = df.ndvi.to_numpy()
    ndvi_mat = ndvi_mat.reshape(data[0].shape[0], data[0].shape[1])

    return ndvi_mat







def deseasonalize_monthly(series):
    """Remove the average seasonal (calendar-month) pattern from a monthly
    series labeled "YYYY-MM"

    inputs:
        - series: time series to be deseasonalized
    outputs:
        - series_des: deseasonalized time series
        - climatology: mean value of each month

    """

    series_des = series.copy(deep = True)  # deep copy: must not mutate the caller's original series

    # compute month climatology
    series_des.rename(lambda x: int(x[5:]), inplace = True)
    climatology = series_des.groupby(series_des.index).mean()

    # deseasonalize series by month climatology
    for i in range(len(series_des)):
        series_des.iloc[i] = series_des.iloc[i] - climatology.iloc[i % 12]

    return series_des, climatology







def mann_kendall_test(series):
    """Mann-Kendall trend test.
    Detects whether later values are consistently ranked higher or lower 
    than earlier ones, more than chance would allow. 

    inputs:
        - series: time series to be tested
    outputs:
        - dict with keys: trend ("increasing"/"decreasing"/"no trend"), s, z,
            p_value, tau (normalized S, in [-1, 1]).
    """


    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 non-NaN observations for a Mann-Kendall test")

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))

    # variance correction for tied values
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)

    if p_value < 0.05 and s > 0:
        trend = "increasing"
    elif p_value < 0.05 and s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return {"trend": trend, "s": int(s), "z": float(z), "p_value": float(p_value), "tau": float(tau)}




def get_climate_data( coords, start_date, end_date, variables):
    """Function that returns the climatic data from the free Open-Meteo Historical Weather API.
    Precipitation-like variables are summed per month; anything else (like temperature) is 
    averaged per month

    inputs:
        - coords: Coordinate box delimiting the square over which to fetch the rainfall data
        - dates: Dates delimiting the time span over which the data is requested. 
        - config: SentinelHub configuration
    outputs:
        - variables : list of variables to be fetched. Can be all variables supported by the 
        Open-Meteo archive API, e.g. "precipitation_sum", "temperature_2m_mean", 
        "temperature_2m_max"... Must be an iterable.
    """



    min_lon, min_lat, max_lon, max_lat = coords
    lat, lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables),
            "timezone": "UTC",
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = pd.DataFrame(resp.json()["daily"])
    daily["period"] = pd.to_datetime(daily["time"]).dt.strftime("%Y-%m")

    monthly = {}
    for var in variables:
        if "precipitation" in var or "rain" in var:
            monthly[var] = daily.groupby("period")[var].sum()
        else:
            monthly[var] = daily.groupby("period")[var].mean()
    return pd.DataFrame(monthly)




def lagged_correlation(ndvi_series, driver_series, labels, driver_labels, max_lag = 3):
    """Pearson correlation between an index series (e.g. mean NDVI) and an
    external driver (e.g. monthly rainfall).

    inputs:
        - ndvi_series: time series of the NDVI index
        - driver_series: time series of the rainfall potentially driving the NDVI variation
        - labels: labels for the ndvi series
        - driver_labels: labels for the river series
        - max_lag: largest lag, in months, to test

    outputs:
        - results: DataFrame with one row per lag (lag_months, r, p_value, n)
        - best: row with the largest |r| correlation value
    """
    driver_labels = driver_labels if driver_labels is not None else labels
    s_ndvi = pd.Series(np.asarray(ndvi_series, dtype=float), index=list(labels))
    s_driver = pd.Series(np.asarray(driver_series, dtype=float), index=list(driver_labels))
    s_driver = s_driver.reindex(s_ndvi.index)

    rows = []
    for lag in range(0, max_lag + 1):
        shifted = s_driver.shift(lag)
        aligned = pd.concat([s_ndvi, shifted], axis=1).dropna()
        if len(aligned) < 3:
            continue
        r, p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        rows.append({"lag_months": lag, "r": r, "p_value": p, "n": len(aligned)})

    results = pd.DataFrame(rows)
    best = None
    if len(results):
        best = results.loc[results["r"].abs().idxmax()]
    return results, best




def phenology_metrics(series, labels):
    """Extract per-year phenology metrics from a monthly index series:
    green-up month, senescence month, season length, and peak timing/value.

    inputs:
        - series: monthly index values
        - labels: labels for the input series
        - labels: labels for the ndvi series
        - driver_labels: labels for the river series
        - max_lag: largest lag, in months, to test

    outputs:
        - DataFrame indexed by year with columns: green_up_month, 
        senescence_month, season_length_months, peak_month, peak_value.
    """


    s = pd.Series(np.asarray(series, dtype=float), index=list(labels))
    years = sorted({lbl.split("-")[0] for lbl in labels})

    rows = []
    for year in years:
        year_labels = [lbl for lbl in labels if lbl.startswith(year)]
        year_vals = s[year_labels]
        if year_vals.isna().all():
            continue

        thr =  (year_vals.min() + year_vals.max()) / 2  # each year uses its own midpoint between that year's min and max as a threshold
        above = (year_vals >= thr).values
        if not above.any():
            continue

        above_idx = np.where(above)[0]
        green_up_idx, senescence_idx = int(above_idx.min()), int(above_idx.max())
        peak_idx = int(np.nanargmax(year_vals.values))

        rows.append({
            "year": year,
            "green_up_month": int(year_labels[green_up_idx].split("-")[1]),
            "senescence_month": int(year_labels[senescence_idx].split("-")[1]),
            "season_length_months": senescence_idx - green_up_idx + 1,
            "peak_month": int(year_labels[peak_idx].split("-")[1]),
            "peak_value": float(year_vals.values[peak_idx]),
        })

    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame(
        columns=["green_up_month", "senescence_month", "season_length_months", "peak_month", "peak_value"]
    )
