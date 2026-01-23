from pathlib import Path
from typing import Dict, List


def _collect_images(root: Path) -> List[str]:
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(root.rglob(ext))
    return [str(p) for p in images]


def scan_nyc_indoor_vpr(root_dir: str) -> List[Dict]:
    root = Path(root_dir)
    zones = []
    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue
        images = _collect_images(scene_dir)
        if not images:
            continue
        zones.append(
            {
                "zone_name": f"NYCIndoorVPR_{scene_dir.name}",
                "description": f"NYC-Indoor-VPR scene {scene_dir.name}",
                "image_paths": images,
            }
        )
    return zones


def scan_openloris_location(root_dir: str, per_location: bool = True) -> List[Dict]:
    root = Path(root_dir)
    if (root / "ourDataset").exists():
        root = root / "ourDataset"
    zones = []
    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue
        if per_location:
            for loc_dir in scene_dir.iterdir():
                if not loc_dir.is_dir():
                    continue
                images = _collect_images(loc_dir)
                if not images:
                    continue
                zones.append(
                    {
                        "zone_name": f"OpenLORISLoc_{scene_dir.name}_{loc_dir.name}",
                        "description": f"OpenLORIS location {scene_dir.name}/{loc_dir.name}",
                        "image_paths": images,
                    }
                )
        else:
            images = _collect_images(scene_dir)
            if not images:
                continue
            zones.append(
                {
                    "zone_name": f"OpenLORISLoc_{scene_dir.name}",
                    "description": f"OpenLORIS scene {scene_dir.name}",
                    "image_paths": images,
                }
            )
    return zones


def scan_cold_subset(root_dir: str) -> List[Dict]:
    root = Path(root_dir)
    zones = []
    for seq_dir in root.iterdir():
        if not seq_dir.is_dir():
            continue
        images = _collect_images(seq_dir)
        if not images:
            continue
        zones.append(
            {
                "zone_name": f"COLD_{seq_dir.name}",
                "description": f"COLD sequence {seq_dir.name}",
                "image_paths": images,
            }
        )
    return zones
