# General imports
from functools import partial
from pathlib import Path
from typing import Optional

import dask.bag as db
import fsspec  # type: ignore
import numpy as np
from dask.distributed import get_client
from loguru import logger
from pandera.typing import DataFrame
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore

from tiling.models import Tile, TileSchema
from tiling.utils import (
    get_array_from_tile,
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    get_stain_vectors_macenko,
    pull_stain_channel,
)


def compute_otsu_score(tile: Tile, slide_path: str, otsu_threshold: float) -> float:
    """
    Return otsu score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_path (str): path to slide
        otsu_threshold (float): otsu threshold value
    """
    tile_arr = get_array_from_tile(tile, slide_path, 10)
    score = np.mean((rgb2gray(tile_arr) < otsu_threshold).astype(int))
    return score


def get_purple_score(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    score = np.mean((r > (g + 10)) & (b > (g + 10)))
    return score


def compute_purple_score(
    tile: Tile,
    slide_path: str,
) -> float:
    """
    Return purple score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
    """
    tile_arr = get_array_from_tile(tile, slide_path, 10)
    return get_purple_score(tile_arr)


def compute_stain_score(
    tile: Tile,
    slide_path: str,
    vectors,
    channel,
    stain_threshold: float,
) -> np.floating:
    """
    Returns stain score for the tile
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
        vectors (np.ndarray): stain vectors
        channel (int): stain channel
        stain_threshold (float): stain threshold value
    """
    tile_arr = get_array_from_tile(tile, slide_path, 10)
    stain = pull_stain_channel(tile_arr, vectors=vectors, channel=channel)
    score = np.mean(stain > stain_threshold)
    return score


def detect_tissue(
    slide_urlpath: str,
    tiles_df: DataFrame[TileSchema],
    output_path: str,
    thumbnail_magnification: Optional[int] = 2,
    filter_query: str = "",
    batch_size: int = 2000,
    storage_options: Optional[dict] = None,
) -> DataFrame[TileSchema]:
    if not storage_options:
        storage_options = {}

    get_client()

    to_mag_scale_factor = get_scale_factor_at_magnification(
        slide_urlpath,
        requested_magnification=thumbnail_magnification,
        storage_options=storage_options,
    )
    logger.info(f"Thumbnail scale factor: {to_mag_scale_factor}")
    # Original thumbnail
    sample_arr = get_downscaled_thumbnail(
        slide_urlpath, to_mag_scale_factor, storage_options=storage_options
    )
    logger.info(f"Sample array size: {sample_arr.shape}")

    with open(Path(output_path) / "sample_arr.png", "wb") as f:
        Image.fromarray(sample_arr).save(f, format="png")

    logger.info("Enhancing image...")
    enhanced_sample_img = ImageEnhance.Contrast(
        ImageEnhance.Color(Image.fromarray(sample_arr)).enhance(10)
    ).enhance(10)
    with open(
        Path(output_path) / "enhanced_sample_arr.png",
        "wb",
    ) as f:
        enhanced_sample_img.save(f, format="png")

    logger.info("HSV space conversion...")
    hsv_sample_arr = np.array(enhanced_sample_img.convert("HSV"))
    with open(
        Path(output_path) / "hsv_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(np.array(hsv_sample_arr)).save(f, "png")

    logger.info("Calculating max saturation...")
    hsv_max_sample_arr = np.max(hsv_sample_arr[:, :, 1:3], axis=2)
    with open(
        Path(output_path) / "hsv_max_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(hsv_max_sample_arr).save(f, "png")

    logger.info("Calculate and filter shadow mask...")
    shadow_mask = np.where(np.max(hsv_sample_arr, axis=2) < 10, 255, 0).astype(np.uint8)
    with open(
        Path(output_path) / "shadow_mask.png",
        "wb",
    ) as f:
        Image.fromarray(shadow_mask).save(f, "png")

    logger.info("Filter out shadow/dust/etc...")
    sample_arr_filtered = np.where(
        np.expand_dims(shadow_mask, 2) == 0, sample_arr, np.full(sample_arr.shape, 255)
    ).astype(np.uint8)
    with open(
        Path(output_path) / "sample_arr_filtered.png",
        "wb",
    ) as f:
        Image.fromarray(sample_arr_filtered).save(f, "png")

    logger.info("Calculating otsu threshold...")
    threshold = threshold_otsu(rgb2gray(sample_arr_filtered))

    logger.info("Calculating stain vectors...")
    stain_vectors = get_stain_vectors_macenko(sample_arr_filtered)

    logger.info("Calculating stain background thresholds...")
    logger.info("Channel 0")
    threshold_stain0 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=0
        ).flatten()
    )
    logger.info("Channel 1")
    threshold_stain1 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=1
        ).flatten()
    )

    # Get the otsu mask
    logger.info("Saving otsu mask")
    otsu_mask = np.where(rgb2gray(sample_arr_filtered) < threshold, 255, 0).astype(
        np.uint8
    )
    with open(Path(output_path) / "otsu_mask.png", "wb") as f:
        Image.fromarray(otsu_mask).save(f, "png")

    logger.info("Saving stain thumbnail")
    deconv_sample_arr = pull_stain_channel(sample_arr_filtered, vectors=stain_vectors)
    with open(
        Path(output_path) / "deconv_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(deconv_sample_arr).save(f, "png")

    logger.info("Saving stain masks")
    stain0_mask = np.where(deconv_sample_arr[..., 0] > threshold_stain0, 255, 0).astype(
        np.uint8
    )
    stain1_mask = np.where(deconv_sample_arr[..., 1] > threshold_stain1, 255, 0).astype(
        np.uint8
    )
    with open(
        Path(output_path) / "stain0_mask.png",
        "wb",
    ) as f:
        Image.fromarray(stain0_mask).save(f, "png")
    with open(
        Path(output_path) / "stain1_mask.png",
        "wb",
    ) as f:
        Image.fromarray(stain1_mask).save(f, "png")

    if filter_query:

        def f_many(iterator, tile_fn, **kwargs):
            return [tile_fn(tile=x, **kwargs) for x in iterator]

        chunks = db.from_sequence(
            tiles_df.itertuples(name="Tile"), partition_size=batch_size
        )

        # calculate scores using cached slide path
        fs, path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)
        simplecache_fs = fsspec.filesystem("simplecache", fs=fs)
        with simplecache_fs.open(path, "rb") as of:
            results = {}
            if "otsu_score" in filter_query:
                logger.info(f"Starting otsu thresholding, threshold={threshold}")

                # chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)
                results["otsu_score"] = chunks.map_partitions(
                    partial(f_many, tile_fn=compute_otsu_score),
                    slide_path=of.name,
                    otsu_threshold=threshold,
                )
            if "purple_score" in filter_query:
                logger.info("Starting purple scoring")
                results["purple_score"] = chunks.map_partitions(
                    partial(f_many, tile_fn=compute_purple_score), slide_path=of.name
                )
            if "stain0_score" in filter_query:
                logger.info(
                    f"Starting stain thresholding, channel=0, threshold={threshold_stain0}"
                )
                results["stain0_score"] = chunks.map_partitions(
                    partial(f_many, tile_fn=compute_stain_score),
                    vectors=stain_vectors,
                    channel=0,
                    stain_threshold=threshold_stain0,
                    slide_path=of.name,
                )
            if "stain1_score" in filter_query:
                logger.info(
                    f"Starting stain thresholding, channel=1, threshold={threshold_stain1}"
                )
                results["stain1_score"] = chunks.map_partitions(
                    partial(f_many, tile_fn=compute_stain_score),
                    vectors=stain_vectors,
                    channel=1,
                    stain_threshold=threshold_stain1,
                    slide_path=of.name,
                )

            for k, v in results.items():
                tiles_df[k] = v.compute()
        logger.info(f"Filtering based on query: {filter_query}")
        tiles_df = tiles_df.query(filter_query)

    logger.info(tiles_df)

    return tiles_df
