import os

from tiling.cli import main


def test_main(tmp_path):
    main([
        "--tile-size", "256",
        "--batch-size", "8",
        "--output-path", str(tmp_path),
        "--otsu-threshold", "0.5",
        "tests/testdata/123.svs"])
    assert os.path.exists(f"{tmp_path}/123.parquet")
