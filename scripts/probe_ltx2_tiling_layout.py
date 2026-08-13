"""Run UPSTREAM's own auto-tiling resolution for the sizes this campaign measured.

No model, no checkpoint: pure ltx_core tiling code at the pinned SHA
fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca.
"""
import sys

sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-pipelines/src")
import torch  # noqa: E402

import ltx_core  # noqa: E402

assert "/home/mudler/_git/LTX-2" in ltx_core.__file__, ltx_core.__file__
from ltx_core.model.video_vae.video_vae import (  # noqa: E402
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
)
from ltx_core.tiling import (  # noqa: E402
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionSizeConfig,
    TileSizeConfig,
    create_tiles,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape  # noqa: E402

SF = SpatioTemporalScaleFactors(time=8, height=32, width=32)
LONG = DimensionSizeConfig(tile_size=768, overlap=64)  # helpers.py:62
FR = DimensionSizeConfig(tile_size=80, overlap=24)  # helpers.py:63


def prepare(latent_shape, cfg):
    """conv_video_decoder.py:359-381 (`ConvVideoDecoder._prepare_tiles`)."""
    splitters = [DEFAULT_SPLIT_OPERATION] * 5
    mappers = [DEFAULT_MAPPING_OPERATION] * 5
    t, h, w = cfg.to_splitters(SF)
    splitters[2], splitters[3], splitters[4] = t, h, w
    if cfg.height.is_tiled():
        mappers[3] = to_mapping_operation(map_spatial_slice, scale=SF.height)
    if cfg.width.is_tiled():
        mappers[4] = to_mapping_operation(map_spatial_slice, scale=SF.width)
    if cfg.frames.is_tiled():
        mappers[2] = to_mapping_operation(map_temporal_slice, scale=SF.time)
    return create_tiles(torch.Size(latent_shape), splitters, mappers)


class _P:
    def __init__(self, f, h, w):
        self.batch, self.frames, self.height, self.width, self.fps = 1, f, h, w, 25.0


for (W, H, F) in [
    (128, 128, 9),
    (320, 192, 25),
    (448, 256, 25),
    (896, 512, 25),
    (1280, 704, 121),
    (1920, 1088, 241),
]:
    cfg = TileSizeConfig.from_long_side(
        long_side=LONG, height=H, width=W, scale_factors=SF, frames=FR
    )
    lat = VideoLatentShape.from_pixel_shape(_P(F, H, W), 128, SF)
    shape = (1, 128, lat.frames, lat.height, lat.width)
    tiles = prepare(shape, cfg)
    groups = group_tiles_by_temporal_slice(tiles)
    full = VideoLatentShape.from_torch_shape(torch.Size(shape)).upscale(SF)
    comp = masks_are_complementary(tiles, full.to_torch_shape())
    peak_frames = max(
        (g[0].out_coords[2].stop - g[0].out_coords[2].start) for g in groups
    )
    peak = peak_frames * 3 * full.height * full.width * 4
    print(
        f"{W}x{H}/{F}f latent={shape[2:]} "
        f"cfg(h={cfg.height.tile_size}/{cfg.height.overlap},"
        f"w={cfg.width.tile_size}/{cfg.width.overlap},"
        f"f={cfg.frames.tile_size}/{cfg.frames.overlap}) "
        f"-> tiles={len(tiles)} groups={len(groups)} complementary={comp} "
        f"chunkbuf={peak / 2**20:.1f} MiB "
        f"fullpix={3 * full.frames * full.height * full.width * 4 / 2**20:.1f} MiB"
    )
