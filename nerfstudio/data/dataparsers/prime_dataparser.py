"""Dataparser SNCF/PRIME railway dataset (robot PRIME, LiDAR Leishen CH128X1, 128 canaux)
pour NeuRAD/SplatAD.

Contrairement à sncf_dataparser.py (écrit pour un banc de test VLP-16, 16 canaux,
élévations hardcodées ±15°), ce dataparser lit l'élévation par ring directement
depuis `beam_inclinations` dans transforms_{split}.json, calculée empiriquement
à l'extraction (voir compute_beam_inclinations dans PRIME_bag_to_splatAD.py).
Ne jamais hardcoder de table d'élévation ici : le CH128X1 a 128 canaux avec un FOV
vertical asymétrique (~-17.7° à +7° mesuré empiriquement), rien à voir avec le VLP16.

Prérequis côté lib nerfstudio (déjà fait) :
  - nerfstudio/data/utils/lidar_elevation_mappings.py : CH128X1_ELEVATION_MAPPING ajouté
  - nerfstudio/cameras/lidars.py : LidarType.CH128X1 ajouté (enum, résolution de noms,
    get_lidar_elevation_mapping, get_lidar_azimuth_resolution, get_lidar_relovution_time)
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

# CH128X1 : divergence de faisceau. Valeurs VLP16 gardées en placeholder si la
# datasheet Leishen n'a pas encore été consultée -- impacte l'antialiasing 3DGS
# (taille effective du gaussien lidar), pas la géométrie des points.
HORIZONTAL_BEAM_DIVERGENCE = 3.49e-3  # 0.2° @10Hz, datasheet CH128X1
VERTICAL_BEAM_DIVERGENCE = 3.89e-3  # 0.223° mesuré (médiane des écarts beam_inclinations)
DATA_FREQUENCY = 20.0  # Hz -- fallback si "time" absent du JSON (mesuré empiriquement sur le bag, dt médian 0.05s)
SCAN_PERIOD = 1.0 / DATA_FREQUENCY  # période de rotation du CH128X1, pour centrer le temps par-point du rolling shutter


def range_view_to_pointcloud(npy: np.ndarray, beam_inclinations: np.ndarray) -> np.ndarray:
    """(H,W,3)=[hit,intensity,range] -> (N,5)=[x,y,z,intensity,t_rel]

    beam_inclinations : array(H,) élévation en radians par ring, lue depuis
    transforms_{split}.json (clé "beam_inclinations"), PAS une constante capteur fixe.

    Convention azimut vérifiée cohérente avec l'encodage dans
    PRIME_bag_to_splatAD.py::pointcloud_to_range_image :
        encode : beta = (pi - atan2(y,x)) % 2pi ; col = round(beta * W/2pi) % W
        decode (ici) : azim = (1 - col/W) * 2pi - pi = pi - col*2pi/W = pi - beta
        -> azim == atan2(y,x) d'origine, cohérent.
    """
    H, W, _ = npy.shape
    assert len(beam_inclinations) == H, (...)
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
    channel_id = row.astype(np.float32)  # ring exact, connu directement (pas d'inférence nécessaire)
    return np.stack([x, y, z, inty, t, channel_id], axis=1).astype(np.float32)


@dataclass
class PRIMEDataParserConfig(ADDataParserConfig):
    """Config dataset ferroviaire SNCF/PRIME -- LiDAR CH128X1."""

    _target: Type = field(default_factory=lambda: PRIMEDataParser)
    data: Path = Path("data/prime")
    sequence: str = "prime"
    cameras: Tuple[str, ...] = ("camera",)
    lidars: Tuple[str, ...] = ("ch128x1",)
    annotation_interval: float = 0.1
    allow_per_point_times: bool = True
    load_cuboids: bool = False  # pas d'annotations bbox dynamiques pour PRIME
    train_split_fraction: float = 0.85
    restrict_azimuth_to_observed_range: Tuple[str, ...] = ("ch128x1",)
    lidar_azimuth_resolution: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"ch128x1": 0.36}  # mesuré : 120° / 332 colonnes utiles
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
    """Dataparser dataset ferroviaire SNCF/PRIME, LiDAR Leishen CH128X1 (128 canaux)."""

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
        # distorsion : images sauvées BRUTES (non undistort) par PRIME_bag_to_splatAD.py,
        # donc k1/k2/p1/p2 doivent être transmis pour que le rendu soit cohérent avec
        # l'image de supervision.
        k1, k2 = float(meta.get("k1", 0.0)), float(meta.get("k2", 0.0))
        p1, p2 = float(meta.get("p1", 0.0)), float(meta.get("p2", 0.0))
        k3 = float(meta.get("k3", 0.0))

        filenames, poses, times = [], [], []
        for i, frame in enumerate(frames):
            filenames.append(self.config.data / frame["file_path"])
            # PRIME_bag_to_splatAD.py écrit toujours transform_matrix -> pas de fallback
            # l2w @ T_xxx nécessaire ici (contrairement à sncf_dataparser.py).
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            poses.append(torch.from_numpy(c2w[:3, :4]).float())
            # temps réel de la frame (secondes, relatif au début de séquence) si dispo,
            # sinon repli synthétique -- le vrai timestamp est important car le sampling
            # lidar PRIME n'est pas parfaitement régulier (dt mesuré: médiane 0.05s, max 0.5s)
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
            pcs = [pc[:, :5] for pc in pcs]  # retire channel_id, inutile hors add_missing_points
    
        return pcs

    def _get_actor_trajectories(self) -> List[Dict]:
        return []  # pas d'acteurs dynamiques annotés pour PRIME