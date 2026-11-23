# Vantyx · AI Side-Scan Sonar Debris & Anomaly Detection (SIH26057)

A working prototype of the system in the SIH-VANTYX-263 deck: it runs the hybrid
**CNN → Mamba → Transformer** model on side-scan sonar tiles and follows the slide 5 workflow:

```
Side-scan sonar ─► Pre-processing ─► AI analysis ─┬─ Known target detection + class confidence
                                                 └─ Unknown anomaly + anomaly score
                 ─► Geolocation ─► Human verification (Confirm / Reject) ─► GIS / Report
```

## Run it

```bash
cd sonar_app
pip install -r requirements.txt
python server.py
```
Open **http://127.0.0.1:8000**. Pick one of the held-out test tiles (or upload any sonar image),
enter the survey navigation (tile-centre lat/lon, heading, swath width, towfish altitude) and press
**Run AI analysis**. The map and basemap load from the internet (Leaflet / CARTO); everything else runs offline.

## What each part does

| Step (slide 5) | Where | How |
|---|---|---|
| Ingest | UI / `POST /api/analyze` | Upload or choose a sample tile, plus navigation metadata |
| Pre-processing | `dataset.preprocess_sonar` | colour palette → intensity (works across sonar devices/palettes), median despeckle, CLAHE gain normalisation, resize to 320 |
| AI analysis – known targets | `hybrid_model.py` | CNN stem (16×) → bidirectional **Mamba** selective state-space block (Mamba-2/SSD form) → **Transformer** self-attention → stride-8 fusion → CenterNet-style heads (class heatmap, box size, offset) |
| AI analysis – unknown anomalies | `pipeline.py` | PaDiM-style: Gaussian model of "normal seabed" features learned from training tiles; Mahalanobis distance → anomaly score 0–1 (0.5 ≈ top-1 % unusual). Strong anomalous regions the class heads did not explain are reported as **UNKNOWN ANOMALY** |
| False-positive diagnosis | `pipeline.py` | biological targets (fish) and low-confidence + seabed-like detections are flagged *Probable False Positive* |
| AI evidence | `pipeline._evidence` | 1. non-natural shape (straight-edge ratio, Hough), 2. strong/structured return (target strength vs seabed), 3. target–shadow consistency (acoustic shadow on the far-range side; shadow length → height estimate) |
| Geolocation | `pipeline.geolocate` | pixel → metres from swath width → rotated by heading → lat/lon (flat-seabed approximation); size in metres |
| Human verification | UI buttons / `POST /api/detections/{id}/review` | Confirm / Reject, persisted in `data/detections.json` |
| GIS / Report | `/api/export/geojson`, `/csv`, `/json`, `/api/report` | GeoJSON opens directly in QGIS/ArcGIS; printable survey report; Leaflet map with debris hotspot heat layer |

## Model & data

* Dataset: `../dtst2` (Roboflow *sonar_detect*, CC BY 4.0) — 405 train / 116 valid / 60 test side-scan tiles,
  classes `aircraft, fish, other, shipwreck`.
* CNN stem = ResNet-18 trunk initialised from ImageNet. For retraining, download
  https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth
  and save it as `weights/resnet18_imagenet.pth` (not needed to run the app); the Mamba block,
  Transformer block, neck and heads are trained from scratch. A fully from-scratch stem was tried first and
  plateaued at ~10 % val mAP (`train_scratch.log`), so the pretrained stem is used.
* Retrain: `python train.py --data ../dtst2 --epochs 40` (≈70 s/epoch on a 2-core CPU, much faster on a GPU).
  Writes `weights/sonar_hybrid.pt` with weights, anomaly statistics, metrics and the training history.

**Current results (40 epochs, CPU)**

| | mAP@0.5 | aircraft | fish | other | shipwreck | object-vs-seabed AP@0.5 |
|---|---|---|---|---|---|---|
| validation (116 tiles) | 41.2 % | 73.6 % | 6.2 % | 7.7 % | 77.1 % | 50.6 % |
| test (60 tiles, held out) | 40.7 % | 93.0 % | 0.0 %* | 11.2 % | 58.8 % | 60.6 % |

\* only 2 fish instances in the test split. Aircraft and shipwreck targets are found reliably; *fish* and *other*
are weak because they are small and rare. Validation mAP was still rising at epoch 40, so longer
training (or a GPU) should help. Every detection still goes through analyst confirm/reject.

## Files
```
server.py        FastAPI backend + exports + report
pipeline.py      preprocessing → detection → anomaly → evidence → geolocation
hybrid_model.py  CNN stem + Mamba block + Transformer block + detection heads
dataset.py       YOLO-label loader, augmentation (sonar gain/speckle/flip), CenterNet targets
train.py         training, mAP evaluation, anomaly-statistics fitting
static/index.html  dashboard
samples/         held-out test tiles for the demo
weights/         trained model
```
