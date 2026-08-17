"""Dataparser SNCF/PRIME railway dataset pour NeuRAD/SplatAD."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, ADDataParserConfig

# Extrinsèque LiDAR→caméra (depuis bag_to_lidargs.py)
T_RGB0_VLP16 = np.linalg.inv(np.array([
    [ 0.0238743541600432, -0.999707744440396,  0.00360642510766516, 0.138922870923538],
    [-0.00736968896588375,-0.00378431903190059,-0.999965147452649, -0.177101909101325],
    [ 0.999687515506770,   0.0238486947027063, -0.00745791352160211,-0.126685267545513],
    [ 0.0, 0.0, 0.0, 1.0]
], dtype=np.float64))

VLP16_INCLINATIONS = np.deg2rad(np.array([
    -15,-13,-11,-9,-7,-5,-3,-1, 1,3,5,7,9,11,13,15
], dtype=np.float32))

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3
VERTICAL_BEAM_DIVERGENCE   = 1.5e-3
DATA_FREQUENCY = 10.0


def range_view_to_pointcloud(npy: np.ndarray) -> np.ndarray:
    """(H,W,3)=[raydrop,intensity,range] → (N,5)=[x,y,z,intensity,t_rel]"""
    H, W, _ = npy.shape
    valid    = (npy[:,:,0] > 0.5) & (npy[:,:,2] > 0.1)
    row, col = np.where(valid)
    r    = npy[row, col, 2]
    inty = npy[row, col, 1]
    elev = VLP16_INCLINATIONS[row]
    azim = (1.0 - col / W) * 2 * np.pi - np.pi
    cos_el = np.cos(elev)
    x = r * cos_el * np.cos(azim)
    y = r * cos_el * np.sin(azim)
    z = r * np.sin(elev)
    t = col.astype(np.float32) / W
    return np.stack([x, y, z, inty, t], axis=1).astype(np.float32)


@dataclass
class SNCFDataParserConfig(ADDataParserConfig):
    """Config dataset ferroviaire SNCF."""
    _target: Type = field(default_factory=lambda: SNCFDataParser)
    data: Path = Path("data/banc_lidargs")
    sequence: str = "banc"
    cameras: Tuple[str, ...] = ("camera",)
    lidars:  Tuple[str, ...] = ("velodyne",)
    annotation_interval: float = 0.1
    allow_per_point_times: bool = True
    load_cuboids: bool = False
    train_split_fraction: float = 0.85


@dataclass
class SNCFDataParser(ADDataParser):
    """Dataparser dataset ferroviaire SNCF/PRIME."""
    config: SNCFDataParserConfig

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
        cx, cy     = float(meta["cx"]),   float(meta["cy"])
        w, h       = int(meta["w"]),      int(meta["h"])

        filenames, poses, times = [], [], []
        for i, frame in enumerate(frames):
            filenames.append(self.config.data / frame["file_path"])
            if "transform_matrix" in frame:
                # Pose caméra directe (frames k>0 sans LiDAR)
                c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            else:
                # Fallback : dériver depuis lidar2world
                l2w = np.array(frame["lidar2world"], dtype=np.float64)
                c2w = l2w @ T_RGB0_VLP16
            # OpenCV → nerfstudio : flip Y et Z
            c2w[:3, 1:3] *= -1
            poses.append(torch.from_numpy(c2w[:3, :4]).float())
            times.append(i / DATA_FREQUENCY)

        cameras = Cameras(
            camera_to_worlds=torch.stack(poses),
            fx=fl_x, fy=fl_y, cx=cx, cy=cy,
            width=w, height=h,
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
            times.append(i / DATA_FREQUENCY)

        lidars = Lidars(
            lidar_to_worlds=torch.stack(poses),
            lidar_type=LidarType.VELODYNE16,
            times=torch.tensor(times, dtype=torch.float64).unsqueeze(-1),
            assume_ego_compensated=False,
            metadata={"sensor_idxs": torch.zeros(len(filenames), 1, dtype=torch.int32)},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )
        return lidars, filenames

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        pcs = []
        for fp in filenames:
            pc = range_view_to_pointcloud(np.load(str(fp)))
            pcs.append(torch.from_numpy(pc).float())
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return pcs

    def _get_actor_trajectories(self) -> List[Dict]:
        return []
