
import numpy as np
import pandas as pd
from scipy import stats

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

    series_des = pd.Series(series)

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
