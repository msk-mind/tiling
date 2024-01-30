import itertools
from typing import Optional

from loguru import logger
from pandera.typing import DataFrame

from tiling.models import Tile, TileSchema
from tiling.utils import (
    coord_to_address,
    get_full_resolution_generator,
    get_scale_factor_at_magnification,
)


def create_manifest(
    slide_urlpath: str,
    tile_size: int,
    requested_magnification: Optional[int] = None,
    storage_options: Optional[dict] = None,
) -> dict:
    """Rasterize a slide into smaller tiles

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    Args:
        slide_urlpath (str): slide url/path
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation

    Returns:
        # DataFrame[TileSchema]: tile manifest
    #"""
    if not storage_options:
        storage_options = {}

    to_mag_scale_factor = get_scale_factor_at_magnification(
        slide_urlpath,
        requested_magnification=requested_magnification,
        storage_options=storage_options,
    )

    if not to_mag_scale_factor % 1 == 0:
        logger.error(f"Bad magnfication scale factor = {to_mag_scale_factor}")
        raise ValueError(
            "You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales"
        )

    full_resolution_tile_size = int(tile_size * to_mag_scale_factor)
    logger.info(
        f"Normalized magnification scale factor for {requested_magnification}x is {to_mag_scale_factor}",
    )
    logger.info(
        f"Requested tile size={tile_size}, tile size at full magnification={full_resolution_tile_size}"
    )

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(
        slide_urlpath,
        tile_size=full_resolution_tile_size,
        storage_options=storage_options,
    )
    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    logger.info(f"tiles x {tile_x_count}, tiles y {tile_y_count}")

    # populate address, coordinates
    tiles = DataFrame[TileSchema](
        [
            Tile(
                address=coord_to_address(address, requested_magnification),
                x_coord=(address[0]) * full_resolution_tile_size,
                y_coord=(address[1]) * full_resolution_tile_size,
                xy_extent=full_resolution_tile_size,
                tile_size=tile_size,
                tile_units="px",
            ).__dict__
            for address in itertools.product(
                range(1, tile_x_count - 1), range(1, tile_y_count - 1)
            )
        ]
    )

    logger.info(f"Number of tiles in raster: {len(tiles)}")

    return tiles
