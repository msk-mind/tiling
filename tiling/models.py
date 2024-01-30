import pandera as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel, ConfigDict


class Tile(BaseModel):
    address: str
    x_coord: int
    y_coord: int
    xy_extent: int
    tile_size: int
    tile_units: str

    model_config = ConfigDict(extra="allow")


class StoredTile(Tile):
    tile_store: str


class LabeledTile(StoredTile):
    Classification: str


class TileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(Tile)
        coerce = True


class StoredTileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(StoredTile)
        coerce = True


class LabeledTileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(LabeledTile)
        coerce = True
