# ValveLens Dataset Registry

This registry records datasets used or staged for ValveLens experiments. It is not a storage location for raw images. Large archives and extracted folders remain local and are ignored by git.

Current graduation-project focus: a small, inspection-first oil/gas/refinery expanded detection dataset. These datasets must not be merged into `data/detection/combined/`, must not overwrite `models/detector.pt`, and must not be used for training until preview and class-balance inspection are complete.

## Final Oil/Gas Expanded Detection Candidates

Prepared output root:

```text
data/detection/oilgas_expanded/
```

Preview and inspection artifacts:

```text
artifacts/detection_oilgas_expanded/
```

Preparation script:

```text
scripts/prepare_oilgas_expanded_dataset.py
```

### Dataset 1: Oil Refinery

- Source: https://universe.roboflow.com/apparatusbeats-olbtp/oil-refinery
- Provider: ApparatusBeats on Roboflow Universe
- License: CC BY 4.0
- Image count: 350
- Original class count: 341
- Media type: unknown until local inspection; refinery/equipment oriented from Roboflow examples
- Task role: refinery-specific macro-equipment detection
- Local download folder: `data_sources/downloads/roboflow/oil_refinery/`
- Local extracted folder: `data_sources/extracted/roboflow/oil_refinery/`
- Prepared output folder: `data/detection/oilgas_expanded/oil_refinery/`
- Artifact folder: `artifacts/detection_oilgas_expanded/oil_refinery/`

Representative original classes:

- `Desalter_base-001`
- `Desalter_pipes-001`
- `Desalter_Tank-001`
- `Heater_beams_VRAY-001`
- `Heater_chasis_VRAY-001`
- `Heater_ducts_VRAY-001`
- `HeatExchanger_Base_001`
- `HeatExchanger_Cylinder_001`
- `HeatExchanger_Pipe_001`
- many additional refinery part classes

Mapped ValveLens classes:

- `desalter`
- `heater`
- `heat_exchanger`
- `pipe`
- `tank_or_cylinder`
- `structure`

Limitations:

- Do not train 341 raw classes.
- Mapping quality must be inspected visually because many class names are part-level CAD or refinery-structure labels.
- This dataset is detection-only; it is not identity validation data.

Recommendation before training: inspect more.

### Dataset 2: Elementos Offshore

- Source: https://universe.roboflow.com/dataset-offshore/elementos-offshore
- Provider: Dataset Offshore on Roboflow Universe
- License: Public Domain
- Image count: 106
- Original class count: 8
- Media type: likely real-image offshore/industrial scenes, verify locally
- Task role: small offshore visual support dataset
- Local download folder: `data_sources/downloads/roboflow/elementos_offshore/`
- Local extracted folder: `data_sources/extracted/roboflow/elementos_offshore/`
- Prepared output folder: `data/detection/oilgas_expanded/elementos_offshore/`
- Artifact folder: `artifacts/detection_oilgas_expanded/elementos_offshore/`

Original classes:

- `equipamento`
- `escada`
- `flange`
- `instrumento`
- `operador`
- `suporte`
- `valvula`
- `vaso`

Mapped ValveLens classes:

- `equipment`
- `ladder`
- `flange`
- `instrument`
- `person`
- `support`
- `valve`
- `vessel`

Limitations:

- Small dataset.
- Some mapped classes are context/support classes rather than final core device classes.
- Useful as visual support, not sufficient alone.

Recommendation before training: inspect more.

### Dataset 3: Object_detection_dataset by Anto

- Source: Roboflow search result: Object_detection_dataset by Anto
- Exact project URL: not confirmed yet
- License: unknown until exact Roboflow page is identified
- Image count: about 1.71k in gauge search result; about 1.25k in valve search result
- Original class count: 4 in the relevant search result
- Media type: unknown until local inspection
- Task role: wellhead, valve, gauge, relay support
- Local download folder: `data_sources/downloads/roboflow/wellhead_valve_gauge/`
- Local extracted folder: `data_sources/extracted/roboflow/wellhead_valve_gauge/`
- Prepared output folder: `data/detection/oilgas_expanded/wellhead_valve_gauge/`
- Artifact folder: `artifacts/detection_oilgas_expanded/wellhead_valve_gauge/`

Known original classes from Roboflow search result:

- `WellHead`
- `gauge`
- `relay`
- `valve`

Mapped ValveLens classes:

- `wellhead`
- `gauge`
- `relay`
- `valve`

Limitations:

- Exact Roboflow project slug is not known yet, so automatic SDK download is not configured.
- Must verify license and image count after staging.
- This is detection-only; it does not validate exact device identity.

Recommendation before training: stage manually or identify exact project URL, then inspect.

### Dataset 4: industrial-multilabel

- Source: https://universe.roboflow.com/yolo-rovw9/industrial-multilabel
- Provider: yolo on Roboflow Universe
- License: MIT
- Image count: 970
- Original class count: 25
- Media type: unknown until local inspection
- Task role: optional broad industrial/factory support dataset
- Local download folder: `data_sources/downloads/roboflow/industrial_multilabel/`
- Local extracted folder: `data_sources/extracted/roboflow/industrial_multilabel/`
- Prepared output folder: `data/detection/oilgas_expanded/industrial_multilabel/`
- Artifact folder: `artifacts/detection_oilgas_expanded/industrial_multilabel/`

Original classes:

- `Laptop`
- `Forklift`
- `Hammer`
- `Screw`
- `Wrench`
- `Rope`
- `safety helmet`
- `clamps`
- `Control Panel`
- `Cooling Towers`
- `cranes`
- `doors`
- `electrical pylon`
- `Gasoline Can`
- `Nuclear reactor`
- `Pebbels`
- `pipes`
- `pliers`
- `Pressure Gauges`
- `Pressure vessel`
- `Screw Driver`
- `Tee Connector`
- `Toolbox`
- `Turbine Generator`
- `Warning signs`

Mapped ValveLens classes:

- `control_panel`
- `cooling_tower`
- `pipe`
- `gauge`
- `pressure_vessel`
- `tee_connector`
- `turbine_generator`
- `warning_sign`
- `clamp`

Ignored classes:

- `Laptop`
- `Forklift`
- `Hammer`
- `Screw`
- `Wrench`
- `Rope`
- `safety helmet`
- `cranes`
- `doors`
- `electrical pylon`
- `Gasoline Can`
- `Nuclear reactor`
- `Pebbels`
- `pliers`
- `Screw Driver`
- `Toolbox`

Limitations:

- Optional dataset only.
- Contains many unrelated factory/tool/object classes.
- Use `--max-images-per-class 200` first to avoid one dataset dominating the expanded experiment.

Recommendation before training: inspect more; include only if selected classes are visually useful.

## Target Expanded Classes

Core oil/gas/refinery classes:

- `valve`
- `gauge`
- `wellhead`
- `flange`
- `instrument`
- `vessel`
- `pipe`
- `desalter`
- `heater`
- `heat_exchanger`
- `structure`
- `tank_or_cylinder`
- `control_panel`
- `relay`

Optional broader industrial classes:

- `pressure_vessel`
- `cooling_tower`
- `tee_connector`
- `turbine_generator`
- `warning_sign`
- `clamp`

Context/support classes from Elementos Offshore:

- `equipment`
- `ladder`
- `person`
- `support`

## Commands

Run these only after each dataset is staged or Roboflow SDK access is available:

```powershell
cd D:\python_works\ValveLens
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset oil_refinery --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset elementos_offshore --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset wellhead_valve_gauge --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset industrial_multilabel --overwrite --max-images-per-class 200
```

## Explicitly Not Main Datasets Now

These may remain in the repo history or local folders, but they are not the current main graduation dataset path:

- Hydraulic Components Detection
- Gas Tube / Flexible Hose datasets
- P&ID or symbol datasets for camera-image detector training

Hydraulic Components Detection can remain as an auxiliary note, but the current focused path is the four-dataset oil/gas/refinery expansion above.
