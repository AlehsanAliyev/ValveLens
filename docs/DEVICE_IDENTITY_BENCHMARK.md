# Device Identity Benchmark

ValveLens needs a small, controlled identity benchmark before v0.3 can be claimed. Public valve/gauge datasets are useful for detection, but they usually do not prove exact device identity because the same physical device instance is not tracked across reference and query images.

This benchmark is deliberately modest. The goal is not to pretend there is a perfect industrial dataset. The goal is to create a repeatable setup that checks whether enrolled devices, reference images, OCR, ReID, fusion, and the uncertainty policy work together.

## Detection Data vs Identity Data

Detection data answers:

- Is there a valve or gauge in the image?
- How well does YOLO localize the object class?

Identity data answers:

- Which exact enrolled device is this?
- Does the query image match the correct reference device?
- Does OCR read the printed ID when a tag is visible?
- Does the system accept or defer appropriately?

Do not treat all valve/gauge dataset images as registered devices. An identity benchmark needs device IDs, reference images, query images, and ground truth.

## What Counts as Controlled Data

Controlled identity data means:

- the same physical object appears in reference and query images
- each device has a stable ID such as `V-1023`, `V-2040`, or `PG-45`
- reference images and query images are separated
- query conditions are recorded, such as clean, low light, glare, blur, occluded, or low contrast
- visible tags are marked explicitly

This can be built with printed tags and home/desk objects for the thesis demo. That is acceptable as a controlled proxy for an industrial identity problem as long as it is described honestly.

## Proxy or Synthetic Identity Data

Proxy identity data may use public images or repeated-looking examples, but only with careful wording:

- call it proxy identity data
- do not claim exact real-world device identity unless the same instance appears in multiple views
- do not mix detector test images into the identity database unless you can justify each device ID
- keep synthetic corruptions as condition tests, not as new device identities

## Proxy Identity Benchmark from Detection Data

The first v0.3 blocker is practical: the device database can contain `V-1023`, `V-2040`, and `PG-45`, but ReID cannot work while the reference folders are empty. Until real repeated photos of physical devices are captured, ValveLens can use a controlled proxy identity benchmark built from the existing YOLO valve/gauge detection dataset.

This proxy workflow:

- crops labeled valve/gauge objects from `data/detection/combined/`
- treats selected crops as pseudo-devices
- assigns stable IDs such as `V-1023`, `V-2040`, and `PG-45`
- generates separate reference and query variants
- adds readable synthetic tags to some images
- creates degraded query conditions such as low light, glare, blur, noise, low contrast, and occlusion
- writes normal `devices_manifest.csv` and `queries_manifest.csv`

This solves the empty-reference problem and allows OCR, ReID, fusion, and decision behavior to be tested end to end. It does not prove final real industrial identity performance. It validates the identity pipeline mechanically and experimentally under controlled generated conditions. Real physical device references are still needed for stronger external validation.

Build the proxy benchmark from the repo root:

```powershell
cd D:\python_works\ValveLens
python .\scripts\build_proxy_device_benchmark.py --devices 3 --refs-per-device 8 --queries-per-device 12 --zone-id <PASTE_REAL_ZONE_ID> --seed 42 --overwrite --easy-tags
python .\scripts\preview_proxy_device_benchmark.py
```

Then enroll and validate from the backend folder:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.enroll_devices_from_manifest --manifest ..\data\device_benchmark\devices_manifest.csv --refs-root ..\data\device_benchmark\refs --force-add-refs
python -m app.cli.rebuild_device_index
python -m app.cli.validate_identity_benchmark --queries-manifest ..\data\device_benchmark\queries_manifest.csv --topk 5 --out ..\artifacts\identity_benchmark
python -m app.cli.smoke_reid --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --topk 5
python -m app.cli.check_ocr_backend
python -m app.cli.smoke_ocr --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --expected V-1023
```

Preview sheets are written to:

```text
artifacts/identity_benchmark/proxy_preview/
```

## Current Proxy Benchmark Interpretation

The proxy benchmark is useful for validating that the identity machinery works:

- devices can be created and enrolled
- reference images can be indexed in the device FAISS store
- ReID can retrieve the expected proxy device from held-out query crops
- OCR can be tested on generated visible tags when the OCR backend is installed
- validation reports can separate missing files, ReID misses, OCR misses, OCR backend failures, and API decision outcomes

Use `--easy-tags` first. It creates larger horizontal high-contrast labels so the first OCR validation checks the pipeline rather than tag legibility. Later, run without `--easy-tags` or with harder degraded tags to study OCR robustness.

If OCR reports no matches while `check_ocr_backend` says Tesseract is missing, that is an environment blocker, not an OCR algorithm failure. Install Tesseract-OCR or configure EasyOCR before reporting OCR exact-match results.

The proxy query images are tight device crops. ReID validation works well on this format, but full API `ACCEPTED` decisions may require full-frame proxy scenes because the runtime pipeline starts with zone localization and detector boxes. If tight crops do not reach `ACCEPTED` through the API, generate full-frame scenes before changing the runtime pipeline.

## Recommended Folder Structure

```text
data/device_benchmark/
  devices_manifest.csv
  queries_manifest.csv
  refs/
    V-1023/
    V-2040/
    PG-45/
  queries/
    V-1023/
      clean/
      low_light/
      glare/
      blur/
      noise/
      low_contrast/
      occluded/
    V-2040/
    PG-45/
```

Reference images under `refs/` should be used for enrollment. Query images under `queries/` should be held out for evaluation.

## Manifests

Device manifest columns:

```text
device_id,type,zone_id,description,has_visible_tag
```

Query manifest columns:

```text
image_path,expected_device_id,expected_type,condition,tag_visible,expected_zone
```

Example manifests are provided:

- `data/device_benchmark/devices_manifest.example.csv`
- `data/device_benchmark/queries_manifest.example.csv`

Copy them to:

- `data/device_benchmark/devices_manifest.csv`
- `data/device_benchmark/queries_manifest.csv`

Then replace `<ZONE_ID>` and image paths with real local values.

## Enrollment

From the backend folder:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.enroll_devices_from_manifest --manifest ..\data\device_benchmark\devices_manifest.csv --refs-root ..\data\device_benchmark\refs
python -m app.cli.rebuild_device_index
```

The enrollment helper:

- creates devices that do not exist
- skips devices that already exist
- adds references from `refs/<device_id>` when the device has no refs yet
- warns about missing folders or missing images
- prints the rebuild command

Use `--force-add-refs` only when you intentionally want to add more reference images to existing devices.

## Validation

Run the manifest validator:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.validate_identity_benchmark --queries-manifest ..\data\device_benchmark\queries_manifest.csv --topk 5 --out ..\artifacts\identity_benchmark
```

Optional API-backed decision validation, if the backend is running:

```powershell
python -m app.cli.validate_identity_benchmark --queries-manifest ..\data\device_benchmark\queries_manifest.csv --topk 5 --backend-url http://localhost:8000 --out ..\artifacts\identity_benchmark
```

Single-image smoke test:

```powershell
python -m app.cli.smoke_reid --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --topk 5
```

Backend tests:

```powershell
pytest app\tests
```

## Metrics to Report

The validator writes:

- `artifacts/identity_benchmark/identity_benchmark_summary.json`
- `artifacts/identity_benchmark/identity_benchmark_summary.csv`

Report:

- total query images
- missing files
- missing expected devices
- ReID top-1 accuracy
- ReID top-k accuracy
- OCR exact-match rate on visible tags
- accepted count if API inference was used
- deferred count if API inference was used
- failure reasons

For interaction sessions, continue using:

```powershell
python -m app.cli.export_metrics --out data\metrics_v03.csv --gt data\gt_sessions.json
python -m app.cli.summarize_metrics --in data\metrics_v03.csv
```

An example session ground-truth file is provided at:

```text
backend/data/gt_sessions.example.json
```

## OCR, ReID, Fusion, and Decision Checks

OCR should be evaluated only when `tag_visible=true`. If the tag is not visible, an OCR miss is not a failure.

ReID should be evaluated for every query image where the expected device exists and the device index is non-empty.

Fusion and decision behavior should be evaluated through the backend API or Live UI. A deferred decision is not automatically a failure. It is correct when evidence is weak, blurry, low-light, ambiguous, or missing enrolled identity.

## Avoid Overclaiming

Do not claim:

- exact industrial identity accuracy from generic valve/gauge detection images
- full robustness from synthetic corruptions alone
- that preprocessing is ready for runtime inference
- that ReID is validated if the device index is empty
- that OCR is validated if no enrolled visible tags were tested

It is fair to claim:

- detection is evaluated on valve/gauge detection data
- zone retrieval is evaluated on public indoor/place data
- robustness preprocessing is an experimental detector-stress test
- identity is evaluated on a small controlled benchmark with enrolled devices

## v0.3 Pass Criteria

Minimum practical pass:

- `devices_count > 0`
- `device_refs_count > 0`
- `device_faiss_size > 0`
- every query image exists
- every expected device exists in DB
- at least one visible tag has an OCR exact match
- at least one device has correct ReID top-k retrieval
- at least one API or Live session reaches `ACCEPTED`
- deferred cases have understandable failure reasons

This makes v0.3 scientifically cleaner: the system can say what it knows, what it does not know, and why.
