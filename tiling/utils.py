from typing import Optional, Tuple, Union

import fsspec
import numpy as np
from loguru import logger
from PIL import Image
from staintools.miscellaneous.get_concentrations import (
    get_concentrations,
)  # type: ignore
from staintools.stain_extraction.macenko_stain_extractor import (
    MacenkoStainExtractor,
)  # type: ignore
from tiffslide import TiffSlide

from tiling.deepzoom import DeepZoomGenerator
from tiling.models import Tile


def get_scale_factor_at_magnification(
    slide_urlpath: str,
    requested_magnification: Optional[int],
    storage_options: Optional[dict] = None,
) -> float:
    """get scale factor at magnification

    Return a scale factor if slide scanned magnification and
    requested magnification are different.

    Args:
        slide (TiffSlide): slide object
        requested_magnification (Optional[int]): requested magnification

    Returns:
        int: scale factor required to achieve requested magnification
    """
    if not storage_options:
        storage_options = {}

    with fsspec.open(slide_urlpath, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        logger.info(f"Slide size = [{slide.dimensions[0]},{slide.dimensions[1]}]")
        # First convert to float to handle true integers encoded as string floats (e.g. '20.000')
        mag_value = float(slide.properties["aperio.AppMag"])

    # Then convert to integer
    scanned_magnification = int(mag_value)

    # # Make sure we don't have non-integer magnifications
    if not int(mag_value) == mag_value:
        raise RuntimeError(
            "Can't handle slides scanned at non-integer magnficiations! (yet)"
        )

    # Verify magnification valid
    scale_factor = 1.0
    if requested_magnification and scanned_magnification != requested_magnification:
        if scanned_magnification < requested_magnification:
            raise ValueError(
                f"Expected magnification <={scanned_magnification} but got {requested_magnification}"
            )
        elif (scanned_magnification % requested_magnification) == 0:
            scale_factor = scanned_magnification // requested_magnification
        else:
            logger.warning("Scale factor is not an integer, be careful!")
            scale_factor = scanned_magnification / requested_magnification

    return scale_factor


def get_full_resolution_generator(
    slide_urlpath: str, tile_size: int, storage_options: Optional[dict] = None
) -> Tuple[DeepZoomGenerator, int]:
    """Return MinimalComputeAperioDZGenerator and generator level

    Args:
        slide_urlpath (str): slide urlpath

    Returns:
        Tuple[MinimalComputeAperioDZGenerator, int]
    """
    if not storage_options:
        storage_options = {}

    generator = DeepZoomGenerator(
        slide_urlpath,
        overlap=0,
        tile_size=tile_size,
        limit_bounds=False,
        storage_options=storage_options,
    )

    generator_level = generator.level_count - 1
    # assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level


def coord_to_address(s: Tuple[int, int], magnification: Optional[int]) -> str:
    """converts coordinate to address

    Args:
        s (tuple[int, int]): coordinate consisting of an (x, y) tuple
        magnification (int): magnification factor

    Returns:
        str: a string consisting of an x_y_z address
    """

    x = s[0]
    y = s[1]
    address = f"x{x}_y{y}"
    if magnification:
        address += f"_z{magnification}"
    return address


def get_array_from_tile(
    tile: Tile,
    slide_urlpath: str,
    size: Optional[int] = None,
    storage_options: Optional[dict] = None,
):
    if not storage_options:
        storage_options = {}
    x, y, extent = tile.x_coord, tile.y_coord, tile.xy_extent
    if size is None:
        resize_size = (tile.tile_size, tile.tile_size)
    else:
        resize_size = (size, size)
    with fsspec.open(slide_urlpath, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        arr = np.array(
            slide.read_region((x, y), 0, (extent, extent)).resize(
                resize_size, Image.NEAREST
            )
        )[:, :, :3]
    return arr


def get_downscaled_thumbnail(
    slide_urlpath: str,
    scale_factor: Union[int, float],
    storage_options: Optional[dict] = None,
) -> np.ndarray:
    if not storage_options:
        storage_options = {}
    """get downscaled thumbnail

    yields a thumbnail image of a whole slide rescaled by a specified scale factor

    Args:
        slide (TiffSlide): slide object
        scale_factor (int): integer scaling factor to resize the whole slide by

    Returns:
        np.ndarray: downsized whole slie thumbnail
    """
    with fsspec.open(slide_urlpath, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        new_width = slide.dimensions[0] // scale_factor
        new_height = slide.dimensions[1] // scale_factor
        img = slide.get_thumbnail((int(new_width), int(new_height)))
        return np.array(img)


def get_stain_vectors_macenko(sample: np.ndarray) -> np.ndarray:
    """get_stain_vectors

    Uses the staintools MacenkoStainExtractor to extract stain vectors

    Args:
        sample (np.ndarray): input patch
    Returns:
        np.ndarray: the stain matrix

    """

    extractor = MacenkoStainExtractor()
    vectors = extractor.get_stain_matrix(sample)
    return vectors


def pull_stain_channel(
    patch: np.ndarray, vectors: np.ndarray, channel: Optional[int] = None
) -> np.ndarray:
    """pull stain channel

    adds 'stain channel' to the image patch

    Args:
        patch (np.ndarray): input image patch
        vectors (np.ndarray): stain vectors
        channel (int): stain channel

    Returns:
        np.ndarray: the input image patch with an added stain channel
    """

    tile_concentrations = get_concentrations(patch, vectors)
    identity = np.array([[1, 0, 0], [0, 1, 0]])
    tmp = 255 * (1 - np.exp(-1 * np.dot(tile_concentrations, identity)))
    tmp = tmp.reshape(patch.shape).astype(np.uint8)
    if channel is not None:
        return tmp[:, :, channel]
    else:
        return tmp
