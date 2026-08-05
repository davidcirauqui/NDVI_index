"""Utility functions for computing the NDVI (Normalized Difference Vegetation
Index) from Sentinel-2 imagery via SentinelHub.
"""

import pandas as pd
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
