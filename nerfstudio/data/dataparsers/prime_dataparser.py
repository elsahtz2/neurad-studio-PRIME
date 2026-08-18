"""Dataparser for the SNCF/PRIME railway dataset (PRIME robot, Leishen CH128X1 LiDAR,
128 channels) for NeuRAD/SplatAD.
 
Per-ring elevation is read directly from `beam_inclinations` in
transforms_{split}.json, computed empirically at extraction time (see
compute_beam_inclinations in PRIME_bag_to_splatAD.py).
Never hardcode an elevation table here: the CH128X1 has 128 channels with an
asymmetric vertical FOV (~-17.7 deg to +7 deg measured empirically), unlike
lower-channel-count sensors whose fixed elevation tables are commonly inlined.
 
Prerequisites on the nerfstudio library side (already done):
  - nerfstudio/data/utils/lidar_elevation_mappings.py: CH128X1_ELEVATION_MAPPING added
  - nerfstudio/cameras/lidars.py: LidarType.CH128X1 added (enum, name resolution,
    get_lidar_elevation_mapping, get_lidar_azimuth_resolution, get_lidar_revolution_time)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, get_lidar_elevation_mapping, transform_points
from nerfstudio.utils import poses as pose_utils
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, ADDataParserConfig

# CH128X1: beam divergence. Affects 3DGS antialiasing (effective LiDAR gaussian
# size), not point geometry.
HORIZONTAL_BEAM_DIVERGENCE = 3.49e-3  # 0.2 deg @10Hz, CH128X1 datasheet
VERTICAL_BEAM_DIVERGENCE = 3.89e-3  # 0.223 deg measured (median of beam_inclinations spacing)
DATA_FREQUENCY = 20.0   # Hz -- fallback when "time" is absent from the JSON (measured empirically on the bag, median dt 0.05 s)
SCAN_PERIOD = 1.0 / DATA_FREQUENCY  # CH128X1 revolution period, used to centre per-point rolling shutter time


def range_view_to_pointcloud(npy: np.ndarray, beam_inclinations: np.ndarray) -> np.ndarray:
    """(H,W,3)=[hit,intensity,range] -> (N,6)=[x,y,z,intensity,t_rel,channel_id]
 
    beam_inclinations: array(H,) per-ring elevation in radians, read from
    transforms_{split}.json (key "beam_inclinations"), NOT a fixed sensor constant.
 
    Azimuth convention verified consistent with the encoding in
    PRIME_bag_to_splatAD.py::pointcloud_to_range_image:
        encode: beta = (pi - atan2(y,x)) % 2pi ; col = round(beta * W/2pi) % W
        decode (here): azim = (1 - col/W) * 2pi - pi = pi - col*2pi/W = pi - beta
        -> azim == original atan2(y,x), consistent.
    """
    H, W, _ = npy.shape
    assert len(beam_inclinations) == H, (
        f"beam_inclinations has {len(beam_inclinations)} entries but the range view has {H} rows"
    )
    valid = (npy[:, :, 0] > 0.5) & (npy[:, :, 2] > 0.1)
    row, col = np.where(valid)
    r = npy[row, col, 2]
    inty = npy[row, col, 1]
    elev = beam_inclinations[row]
    azim = (1.0 - col / W) * 2 * np.pi - np.pi
    cos_el = np.cos(elev)
    x = r * cos_el * np.cos(azim)
    y = r * cos_el * np.sin(azim)
    z = r * np.sin(elev)
    t = (col.astype(np.float32) / W - 0.5) * SCAN_PERIOD
    channel_id = row.astype(np.float32)  # exact ring, known directly (no inference needed)
    return np.stack([x, y, z, inty, t, channel_id], axis=1).astype(np.float32)


@dataclass
class PRIMEDataParserConfig(ADDataParserConfig):
    """Config for the SNCF/PRIME railway dataset -- CH128X1 LiDAR."""

    _target: Type = field(default_factory=lambda: PRIMEDataParser)
    data: Path = Path("data/prime")
    sequence: str = "prime"
    cameras: Tuple[str, ...] = ("camera",)
    lidars: Tuple[str, ...] = ("ch128x1",)
    annotation_interval: float = 0.1
    allow_per_point_times: bool = True
    load_cuboids: bool = False  # no dynamic bounding box annotations for PRIME for now
    train_split_fraction: float = 0.85
    restrict_azimuth_to_observed_range: Tuple[str, ...] = ("ch128x1",)
    lidar_azimuth_resolution: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"ch128x1": 0.36}  # measured: 120 deg / 332 usable columns
    )
    lidar_elevation_mapping: Optional[Dict[str, Dict[int, float]]] = field(
        default_factory=lambda: {"ch128x1": get_lidar_elevation_mapping(LidarType.CH128X1)}
    )
    skip_elevation_channels: Optional[Dict[str, Tuple[int, ...]]] = field(
        default_factory=lambda: {"ch128x1": ()}
    )
    add_missing_points: bool = True


@dataclass
class PRIMEDataParser(ADDataParser):
    """Dataparser for the SNCF/PRIME railway dataset, Leishen CH128X1 LiDAR (128 channels)."""

    config: PRIMEDataParserConfig

    def _load_frames(self, split: str) -> tuple:
        path = self.config.data / f"transforms_{split}.json"
        if not path.exists():
            path = self.config.data / "transforms_train.json"
        with open(path) as f:
            meta = json.load(f)
        return meta, meta["frames"]

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        meta, frames = self._load_frames("train")
        fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
        cx, cy = float(meta["cx"]), float(meta["cy"])
        w, h = int(meta["w"]), int(meta["h"])
        # distortion: images are saved RAW (not undistorted) by PRIME_bag_to_splatAD.py,
        # so k1/k2/p1/p2 must be passed through for the render to stay consistent with
        # the supervision image.
        k1, k2 = float(meta.get("k1", 0.0)), float(meta.get("k2", 0.0))
        p1, p2 = float(meta.get("p1", 0.0)), float(meta.get("p2", 0.0))
        k3 = float(meta.get("k3", 0.0))

        filenames, poses, times = [], [], []
        for i, frame in enumerate(frames):
            filenames.append(self.config.data / frame["file_path"])
            # PRIME_bag_to_splatAD.py always writes transform_matrix, already expressed
            # in world coordinates. No composition with an intermediate transform is
            # needed here, the matrix is used as read.
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            poses.append(torch.from_numpy(c2w[:3, :4]).float())
            # actual frame time (seconds, relative to sequence start) when available,
            # otherwise a synthetic fallback. The real timestamp matters because PRIME
            # LiDAR sampling is not perfectly regular (measured dt: median 0.05 s, max 0.5 s)
            times.append(frame.get("time", i / DATA_FREQUENCY))

        cameras = Cameras(
            camera_to_worlds=torch.stack(poses),
            fx=fl_x,
            fy=fl_y,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            distortion_params=torch.tensor([k1, k2, k3, 0.0, p1, p2], dtype=torch.float32),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(times, dtype=torch.float64).unsqueeze(-1),
            metadata={"sensor_idxs": torch.zeros(len(frames), 1, dtype=torch.int32)},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        meta, frames = self._load_frames("train")
        poses, times, filenames = [], [], []
        for i, frame in enumerate(frames):
            if "lidar_file_path" not in frame:
                continue
            filenames.append(self.config.data / frame["lidar_file_path"])
            l2w = np.array(frame["lidar2world"], dtype=np.float64)
            poses.append(torch.from_numpy(l2w[:3, :4]).float())
            times.append(frame.get("time", i / DATA_FREQUENCY))

        lidars = Lidars(
            lidar_to_worlds=torch.stack(poses),
            lidar_type=LidarType.CH128X1,
            times=torch.tensor(times, dtype=torch.float64).unsqueeze(-1),
            assume_ego_compensated=False,
            metadata={"sensor_idxs": torch.zeros(len(filenames), 1, dtype=torch.int32)},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )
        self._beam_inclinations = np.array(meta["beam_inclinations"], dtype=np.float32)
        return lidars, filenames

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        pcs = []
        for fp in filenames:
            pc = range_view_to_pointcloud(np.load(str(fp)), self._beam_inclinations)
            pcs.append(torch.from_numpy(pc).float())  # [N,6] x,y,z,intensity,t_rel,channel_id
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
    
        if self.config.add_missing_points:
            poses = lidars.lidar_to_worlds
            times = lidars.times.squeeze(-1)
            missing_points = []
            for point_cloud, l2w, time in zip(pcs, poses, times):
                pc = point_cloud.clone().double()
                pc[:, 4] = pc[:, 4] + time
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc, poses, times)
                pc[:, 4] = point_cloud[:, 4].clone()
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                miss_pc = self._get_missing_points(pc, interpolated_poses, "ch128x1")
                miss_pc = miss_pc[..., [0, 1, 2, 4, 5, 3]]
                missing_points.append(miss_pc.float())
            pcs = [
                torch.cat([pc[:, :5], missing[:, :5]], dim=0).float()
                for pc, missing in zip(pcs, missing_points)
            ]
        else:
            pcs = [pc[:, :5] for pc in pcs]  # drop channel_id, unused outside add_missing_points
    
        return pcs

    def _get_actor_trajectories(self) -> List[Dict]:
        return []  # no annotated dynamic actors for PRIME