"""
rellis_dataparser.py  —  Dataparser SplatAD pour RELLIS-3D (Ouster OS1-64)
Format produit par bag_to_splatAD_rellis.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, ADDataParserConfig, OPENCV_TO_NERFSTUDIO
# range_view_to_pointcloud non utilisé pour RELLIS (format xyz direct)

# ─── Constantes caméra Basler acA1920 ────────────────────────────────────────
IMG_W, IMG_H = 1920, 1200
FX, FY       = 2813.643, 2808.326
CX, CY       = 969.286, 624.050
DATA_FREQUENCY = 10.0  # Hz

# Divergence de beam (valeurs typiques OS1-64)
HORIZONTAL_BEAM_DIVERGENCE = 0.003  # rad
VERTICAL_BEAM_DIVERGENCE   = 0.003  # rad


@dataclass
class RellisDataParserConfig(ADDataParserConfig):
    """Config pour le dataparser RELLIS."""
    _target: type = field(default_factory=lambda: RellisDataParser)
    data: Path = Path("data/rellis")
    annotation_interval: float = 0.1  # 1/10Hz
    cameras: Tuple[str, ...] = ("pylon_camera",)
    lidars:  Tuple[str, ...] = ("os1_lidar",)


@dataclass
class RellisDataParser(ADDataParser):
    """Dataparser pour RELLIS-3D."""
    config: RellisDataParserConfig

    def _load_meta(self):
        with open(self.config.data / 'meta.json') as f:
            meta = json.load(f)
        with open(self.config.data / 'lidar_poses.json') as f:
            lidar_poses_raw = json.load(f)
        return meta, lidar_poses_raw

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        meta, lidar_poses_raw = self._load_meta()

        T_base_cam   = np.array(meta['T_base_cam'])
        T_base_lidar = np.array(meta['T_base_lidar'])
        T_lidar_cam  = np.linalg.inv(T_base_lidar) @ T_base_cam

        frame_ids = sorted(int(k) for k in lidar_poses_raw.keys())
        filenames, poses, times = [], [], []

        # OpenCV → OpenGL/nerfstudio : flip Y et Z des axes colonnes
        for i, fid in enumerate(frame_ids):
            T_world_lidar = np.array(lidar_poses_raw[str(fid)])
            T_world_cam   = T_world_lidar @ T_lidar_cam
            # OpenCV → nerfstudio : appliquer sur R seulement (comme PandaSet)
            T_world_cam[:3, :3] = T_world_cam[:3, :3] @ OPENCV_TO_NERFSTUDIO

            filenames.append(self.config.data / 'images' / f'{fid:06d}.png')
            poses.append(torch.from_numpy(T_world_cam[:3, :4]).float())
            times.append(i / DATA_FREQUENCY)

        cameras = Cameras(
            camera_to_worlds=torch.stack(poses),
            fx=FX, fy=FY, cx=CX, cy=CY,
            width=IMG_W, height=IMG_H,
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(times, dtype=torch.float64).unsqueeze(-1),
            metadata={"sensor_idxs": torch.zeros(len(frame_ids), 1, dtype=torch.int32)},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        meta, lidar_poses_raw = self._load_meta()

        frame_ids = sorted(int(k) for k in lidar_poses_raw.keys())
        poses, times, filenames = [], [], []

        for i, fid in enumerate(frame_ids):
            T_world_lidar = np.array(lidar_poses_raw[str(fid)])
            poses.append(torch.from_numpy(T_world_lidar[:3, :4]).float())
            times.append(i / DATA_FREQUENCY)
            filenames.append(self.config.data / 'lidar' / f'{fid:06d}.npy')

        lidars = Lidars(
            lidar_to_worlds=torch.stack(poses),
            lidar_type=LidarType.OUSTER64,
            times=torch.tensor(times, dtype=torch.float64).unsqueeze(-1),
            assume_ego_compensated=False,
            metadata={"sensor_idxs": torch.zeros(len(filenames), 1, dtype=torch.int32)},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )
        return lidars, filenames

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        """Lit les fichiers .npy (N,4)=[x,y,z,intensity] et retourne (N,5)=[x,y,z,intensity,t_rel] en sensor frame."""
        pcs = []
        for fp in filenames:
            pts = np.load(str(fp)).astype(np.float32)  # (N, 4): x, y, z, intensity
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            intensity = pts[:, 3]
            # t_rel depuis azimuth [0, 1]
            azim = np.arctan2(y, x)
            t_rel = (azim + np.pi) / (2 * np.pi)
            # Points déjà en sensor frame (repère lidar local)
            pc = np.stack([x, y, z, intensity, t_rel], axis=1)
            pcs.append(torch.from_numpy(pc).float())
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return pcs

    def _get_actor_trajectories(self):
        return []
