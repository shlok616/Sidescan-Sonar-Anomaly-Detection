"""
Vantyx - Side-Scan Sonar Debris & Anomaly Detection: FastAPI backend.

    pip install -r requirements.txt
    python server.py            ->  open http://127.0.0.1:8000
"""
import csv
import io
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hybrid_model import CLASS_NAMES
from pipeline import SonarAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("SONAR_WEIGHTS", os.path.join(HERE, "weights", "sonar_hybrid.pt"))
SAMPLES = os.path.join(HERE, "samples")
STORE = os.path.join(HERE, "data", "detections.json")

app = FastAPI(title="Vantyx Sonar AI")
analyzer = SonarAnalyzer(WEIGHTS)
lock = threading.Lock()


def load_store():
    if os.path.exists(STORE):
        with open(STORE) as f:
            return json.load(f)
    return {"scans": [], "detections": []}


def save_store(s):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w") as f:
        json.dump(s, f, indent=1)


store = load_store()


def public(d):
    return {k: v for k, v in d.items() if k != "crop"}


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/model")
def model_info():
    m = analyzer.metrics
    return {"name": "Hybrid CNN → Mamba (SSD) → Transformer detector", "classes": CLASS_NAMES,
            "params_m": round(sum(p.numel() for p in analyzer.model.parameters()) / 1e6, 2),
            "metrics": m, "history": analyzer.history, "device": str(analyzer.device)}


@app.get("/api/samples")
def samples():
    return sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".png", ".jpeg"))) if os.path.isdir(SAMPLES) else []


@app.get("/samples/{name}")
def sample_file(name: str):
    p = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(None), sample: str = Form(None),
                  lat: float = Form(12.9716), lon: float = Form(77.5946), heading: float = Form(0.0),
                  swath_m: float = Form(100.0), altitude: float = Form(10.0), conf: float = Form(0.3)):
    if file is not None and file.filename:
        raw, name = await file.read(), file.filename
    elif sample:
        p = os.path.join(SAMPLES, os.path.basename(sample))
        if not os.path.exists(p):
            raise HTTPException(404, "sample not found")
        raw, name = open(p, "rb").read(), os.path.basename(p)
    else:
        raise HTTPException(400, "upload an image or choose a sample")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    t0 = time.time()
    nav = {"lat": lat, "lon": lon, "heading": heading, "swath_m": swath_m, "altitude": altitude}
    res = analyzer.analyze(img, nav, conf_thresh=conf)
    scan_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with lock:
        start = len(store["detections"])
        for k, d in enumerate(res["detections"]):
            d.update({"id": f"DET-{start + k + 1:04d}", "scan_id": scan_id, "source": name, "time": now,
                      "review": None, "reviewer_note": ""})
            store["detections"].append(public(d))
        store["scans"].append({"scan_id": scan_id, "source": name, "time": now, "nav": nav,
                               "n_detections": len(res["detections"])})
        save_store(store)
    res.update({"scan_id": scan_id, "source": name, "nav": nav, "elapsed_ms": int((time.time() - t0) * 1000)})
    return res


class Review(BaseModel):
    decision: str  # "confirmed" | "rejected" | null
    note: str = ""


@app.post("/api/detections/{det_id}/review")
def review(det_id: str, r: Review):
    if r.decision not in ("confirmed", "rejected", "pending"):
        raise HTTPException(400, "decision must be confirmed / rejected / pending")
    with lock:
        for d in store["detections"]:
            if d["id"] == det_id:
                d["review"] = None if r.decision == "pending" else r.decision
                d["reviewer_note"] = r.note
                d["status"] = {"confirmed": "Confirmed", "rejected": "Rejected"}.get(r.decision, "Needs Verification")
                d["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                save_store(store)
                return d
    raise HTTPException(404)


@app.get("/api/detections")
def detections():
    return store["detections"]


@app.delete("/api/detections")
def clear():
    with lock:
        store["detections"].clear(); store["scans"].clear(); save_store(store)
    return {"ok": True}


def geojson():
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
              "properties": {k: v for k, v in d.items() if k not in ("lat", "lon", "evidence")} |
              {"evidence": "; ".join(e["name"] for e in d["evidence"] if e["present"])}} for d in store["detections"]]
    return {"type": "FeatureCollection", "name": "vantyx_sonar_detections", "features": feats}


@app.get("/api/export/{fmt}")
def export(fmt: str):
    if fmt == "geojson":
        return Response(json.dumps(geojson(), indent=1), media_type="application/geo+json",
                        headers={"Content-Disposition": "attachment; filename=sonar_detections.geojson"})
    if fmt == "json":
        return Response(json.dumps(store, indent=1), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=sonar_detections.json"})
    if fmt == "csv":
        cols = ["id", "source", "time", "kind", "class", "target", "category", "confidence", "anomaly_score",
                "lat", "lon", "length_m", "width_m", "height_m", "status", "review", "reviewer_note", "evidence"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        for d in store["detections"]:
            w.writerow([d.get("id"), d.get("source"), d.get("time"), d.get("kind"), d.get("class"), d.get("target"),
                        d.get("category"), d.get("confidence"), d.get("anomaly_score"), d.get("lat"), d.get("lon"),
                        d["size_m"][0], d["size_m"][1], d.get("height_m"), d.get("status"), d.get("review"),
                        d.get("reviewer_note"), "; ".join(e["name"] for e in d["evidence"] if e["present"])])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sonar_detections.csv"})
    raise HTTPException(404)


def fmt_conf(d):
    return "" if d["confidence"] is None else f"{d['confidence'] * 100:.0f}%"


@app.get("/api/report", response_class=HTMLResponse)
def report():
    ds = store["detections"]
    conf = [d for d in ds if d.get("review") == "confirmed"]
    rej = [d for d in ds if d.get("review") == "rejected"]
    pend = [d for d in ds if not d.get("review")]
    rows = "".join(
        f"<tr><td>{d['id']}</td><td>{d['source'][:28]}</td><td>{d['target']}</td><td>{d['category']}</td>"
        f"<td>{fmt_conf(d)}</td><td>{d['anomaly_score']:.2f}</td>"
        f"<td>{d['lat']:.5f}, {d['lon']:.5f}</td><td>{d['size_m'][0]} × {d['size_m'][1]} m</td>"
        f"<td>{d['status']}</td><td>{', '.join(e['name'] for e in d['evidence'] if e['present']) or '-'}</td></tr>"
        for d in ds)
    m = analyzer.metrics.get("test", {})
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Sonar Survey Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#111}}h1{{margin:0}}table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{border:1px solid #ccc;padding:5px 7px;text-align:left}}th{{background:#0b3d2e;color:#fff}}.k{{display:inline-block;margin:10px 18px 10px 0}}
.k b{{font-size:22px;display:block}}small{{color:#555}}</style></head><body>
<h1>Side-Scan Sonar Debris &amp; Anomaly Survey Report</h1>
<small>Team Vantyx · SIH26057 · generated {datetime.now().strftime('%d %b %Y %H:%M')}</small>
<div><span class=k><b>{len(store['scans'])}</b>sonar tiles analysed</span><span class=k><b>{len(ds)}</b>AI detections</span>
<span class=k><b>{len(conf)}</b>confirmed</span><span class=k><b>{len(rej)}</b>rejected</span><span class=k><b>{len(pend)}</b>awaiting review</span></div>
<p><small>Model: hybrid CNN→Mamba→Transformer detector. Held-out test mAP@0.5 = {m.get('mAP50', 0):.2f},
class-agnostic AP@0.5 = {m.get('AP50_class_agnostic', 0) or 0:.2f}. All detections are AI suggestions; only analyst-confirmed
items should be treated as verified. Positions use flat-seabed georeferencing from the entered survey navigation.</small></p>
<table><tr><th>ID</th><th>Source tile</th><th>Target</th><th>Category</th><th>Conf.</th><th>Anomaly</th><th>Position</th><th>Size</th><th>Status</th><th>AI evidence</th></tr>{rows}</table>
<script>window.onload=()=>setTimeout(()=>window.print(),400)</script></body></html>"""


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Vantyx Sonar AI running → http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
