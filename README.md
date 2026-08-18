# neurad-studio-PRIME

Fork of [neurad-studio](https://github.com/georghess/neurad-studio) adapted for the
SNCF PRIME railway maintenance robot, which carries a Leishen CH128X1 LiDAR
(128 channels) and a MER2-302-56U3C global shutter camera. 


## Modifications

| File | Change |
|---|---|
| `nerfstudio/data/dataparsers/prime_dataparser.py` | New. Dataparser for the PRIME dataset format. |
| `nerfstudio/cameras/lidars.py` | Added `LidarType.CH128X1` and its elevation mapping, azimuth resolution and revolution time. |
| `nerfstudio/data/utils/lidar_elevation_mappings.py` | Added `CH128X1_ELEVATION_MAPPING`. |
| `nerfstudio/configs/dataparser_configs.py` | Registered `prime-data`. |
| `nerfstudio/data/dataparsers/ad_dataparser.py` | Added `restrict_azimuth_to_observed_range`. Missing-point reconstruction assumed 360 deg coverage, which invents points outside the physical field of view of a restricted-FOV sensor. |

## Installation

Three git-based dependencies were removed from `pyproject.toml` and must be
installed manually.

```bash
pip install -e . --no-deps
pip install viser
git clone https://github.com/carlinds/splatad.git gsplat-splatad
pip install -e ./gsplat-splatad
```

`gsplat` must come from the SplatAD fork, not from PyPI. The PyPI package lacks the
LiDAR rasterization kernels.

## The PRIME dataparser

Per-ring elevation is not hardcoded. It is computed empirically at extraction time. The CH128X1 has
an asymmetric vertical field of view, measured at approximately -17.7 deg to +7 deg,
so an evenly spaced elevation table would misplace every point.

Images are stored raw rather than undistorted. Distortion coefficients are passed
through to the camera model so that renders stay consistent with the supervision
images.

Frame timestamps are read from the JSON when present. PRIME LiDAR sampling is not
perfectly regular, with a measured median interval of 0.05 s and gaps up to 0.5 s.

## Usage

This fork is not meant to be run on its own. Dataset extraction and training are
driven from the parent repository
[SplatAD_for_PRIME](https://github.com/elsahtz2/SplatAD_for_PRIME), through
`PRIME_bag_to_splatAD.py` and `run_SplatAD_prime.py`.

General nerfstudio documentation lives in the upstream repository.

## Attribution

- **NeuRAD** — Hess et al., *NeuRAD: Neural Rendering for Autonomous Driving*,
  CVPR 2024. https://github.com/georghess/neurad-studio
- **SplatAD** — Lindström et al., *SplatAD: Real-Time LiDAR and Camera Rendering
  with 3D Gaussian Splatting for Autonomous Driving*.
  https://github.com/carlinds/splatad

Upstream code is licensed under Apache License 2.0. The modifications listed above
were made by the author. Original licence and notice files are retained unchanged.

Work carried out during Elsa Heitz Master's thesis internship at SNCF, February to August 2026.
