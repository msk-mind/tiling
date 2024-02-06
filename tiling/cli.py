from pathlib import Path

import configargparse
import yaml
from dask.distributed import Client

from tiling.tiler import create_manifest
from tiling.tissue_detection import detect_tissue


def main(argv=None):
    p = configargparse.ArgParser(
        description="Create a tile manifest for a slide",
    )
    p.add("-c", "--my-config", is_config_file=True, help="config file path")
    p.add("-o", "--output-path", required=True, type=Path, help="output path")
    p.add("-ts", "--tile-size", nargs="?", type=int, default=512, help="tile size")
    p.add("--slide-id", nargs="?", type=str, help="slide ID")
    p.add("--otsu-threshold", nargs="?", type=float, help="otsu score threshold")
    p.add("--purple-threshold", nargs="?", type=float, help="purple score threshold")
    p.add("--stain0-threshold", nargs="?", type=float, help="stain 0 threshold")
    p.add("--stain1-threshold", nargs="?", type=float, help="stain 1 threshold")
    p.add("-tm", "--tile-magnification", nargs="?", type=int, help="tile magnification")
    p.add("--custom-filter", nargs="?", type=str, help="custom filter string")
    p.add("-bs", "--batch-size", nargs="?", type=int, default=2000, help="batch size")
    p.add("--storage-options", type=yaml.safe_load)
    p.add("-nc", "--num-cores", nargs="?", type=int, help="number of cores")

    p.add("slide_urlpath", type=str, help="slide url/path")

    args = p.parse_args(argv)

    dask_options = {}
    if args.num_cores:
        dask_options = {
            "n_workers": 1,
            "threads_per_worker": args.num_cores,
        }

    Client(**dask_options)

    storage_options = {}
    if args.storage_options:
        storage_options = args.storage_options

    slide_id = Path(args.slide_urlpath).stem
    if args.slide_id:
        slide_id = args.slide_id

    tiles_df = create_manifest(
        args.slide_urlpath,
        requested_magnification=args.tile_magnification,
        tile_size=args.tile_size,
        storage_options=storage_options,
    )
    if (
        args.custom_filter
        or args.otsu_threshold
        or args.purple_threshold
        or args.stain0_threshold
        or args.stain1_threshold
    ):
        filter_query = ""
        if args.custom_filter:
            filter_query = args.custom_filter
        else:
            filter_strs = []
            if args.otsu_threshold:
                filter_strs.append(f"otsu_score > {args.otsu_threshold}")
            if args.purple_threshold:
                filter_strs.append(f"purple_score > {args.purple_threshold}")
            if args.stain0_threshold:
                filter_strs.append(f"stain0_score > {args.stain0_threshold}")
            if args.stain1_threshold:
                filter_strs.append(f"stain1_score > {args.stain1_threshold}")
            filter_query = " and ".join(filter_strs)
        tiles_df = detect_tissue(
            args.slide_urlpath,
            tiles_df,
            args.output_path,
            filter_query=filter_query,
            batch_size=args.batch_size,
            storage_options=storage_options,
        )

    tiles_path = args.output_path / f"{slide_id}.parquet"

    tiles_df.to_parquet(tiles_path)


if __name__ == "__main__":
    main()
