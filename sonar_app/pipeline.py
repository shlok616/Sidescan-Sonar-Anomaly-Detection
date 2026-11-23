"""
End-to-end inference pipeline (slide 5 "Implementation Workflow"):

  Side-scan sonar -> Pre-processing -> AI analysis
        |-- Known target detection + class confidence
        |-- Unknown anomaly + anomaly score
  -> Geolocation -> (human verification, in server.py) -> GIS / report
"""
import base64
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import IMG_SIZE, preprocess_sonar
from hybrid_model import CLASS_NAMES, HybridSonarDetector, decode, iou

# How each dataset class maps onto the debris / anomaly taxonomy used in the report
CLASS_INFO = {
    "aircraft": {"label": "AIRCRAFT WRECKAGE", "category": "Debris (man-made)", "man_made": True},
    "shipwreck": {"label": "SHIPWRECK", "category": "Debris (man-made)", "man_made": True},
    "other": {"label": "UNIDENTIFIED OBJECT", "category": "Anomaly (unclassified)", "man_made": None},
    "fish": {"label": "FISH / BIOLOGICAL", "category": "Natural (likely false positive)", "man_made": False},
}


def _b64(img, ext=".jpg", q=88):
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else [])
    return f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64," + base64.b64encode(buf).decode()


def _overlay(gray, heat01, alpha=0.5):
    h = cv2.resize(heat01.astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap(np.uint8(np.clip(h, 0, 1) * 255), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, col, alpha, 0)


class SonarAnalyzer:
    def __init__(self, weights_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model = HybridSonarDetector().to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        a = ck["anomaly"]
        self.mu, self.icov = a["mu"].to(self.device), a["icov"].to(self.device)
        self.p50, self.p99 = a["p50"], a["p99"]
        self.metrics = ck.get("metrics", {})
        self.history = ck.get("history", [])

    # ------------------------------------------------------------------ AI analysis
    @torch.no_grad()
    def _infer(self, x):
        t = torch.from_numpy(x)[None, None].to(self.device)
        out = self.model(t, return_features=True)
        f = out["features"][0].permute(1, 2, 0)                     # (H,W,C)
        d = f - self.mu
        maha = torch.sqrt(((d @ self.icov) * d).sum(-1).clamp(min=0))
        # 0.5 == as unusual as the top 1 % of training seabed patches
        anom = 1 - torch.exp(-math.log(2) * (maha - self.p50).clamp(min=0) / (self.p99 - self.p50))
        anom = F.avg_pool2d(anom[None, None], 3, 1, 1)[0, 0]
        act = out["activation"][0, 0]
        act = (act - act.min()) / (act.max() - act.min() + 1e-6)
        dets = decode(out, IMG_SIZE, thresh=0.2)[0]
        return dets, anom.cpu().numpy(), act.cpu().numpy(), out["heatmap"].sigmoid()[0].cpu().numpy()

    # ------------------------------------------------------------------ AI evidence
    @staticmethod
    def _evidence(gray, box, nadir_x, mpp, altitude):
        H, W = gray.shape
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        bw, bh = max(2, x2 - x1), max(2, y2 - y1)
        roi = gray[y1:y2, x1:x2]
        pad = int(0.6 * max(bw, bh))
        X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
        ring = gray[Y1:Y2, X1:X2].astype(np.float32).copy()
        ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = np.nan
        bg = ring[~np.isnan(ring)]
        bg_med, bg_std = (float(np.median(bg)), float(bg.std() + 1e-3)) if bg.size else (float(gray.mean()), float(gray.std() + 1e-3))

        # 1. Non-natural shape: fraction of edge pixels lying on long straight segments
        edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 60, 160)
        n_edge = int(edges.sum() / 255)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=max(8, int(0.12 * min(bw, bh))), maxLineGap=4)
        line_len = sum(math.hypot(l[2] - l[0], l[3] - l[1]) for l in np.asarray(lines).reshape(-1, 4)) if lines is not None else 0
        straight = min(1.0, line_len / max(n_edge, 1))
        # 2. Target strength / structured texture: bright returns vs. surrounding seabed
        strength = (float(np.percentile(roi, 95)) - bg_med) / bg_std if roi.size else 0
        edge_density = n_edge / max(roi.size, 1)
        # 3. Target-shadow consistency: an acoustic shadow (dark, low-return zone) should lie directly
        #    behind the target, on the far-range side. Tiles may be cropped/rotated, so test all four
        #    sides and keep the darkest; prefer the side facing away from nadir when it is close.
        sw, sh = max(3, int(0.6 * bw)), max(3, int(0.6 * bh))
        sides = {
            "starboard": (gray[y1:y2, x2:min(W, x2 + sw)], 1, 0),
            "port": (gray[y1:y2, max(0, x1 - sw):x1], -1, 0),
            "aft": (gray[y2:min(H, y2 + sh), x1:x2], 0, 1),
            "fore": (gray[max(0, y1 - sh):y1, x1:x2], 0, -1),
        }
        contrast = {k: ((bg_med - float(v[0].mean())) / bg_std if v[0].size else -9) for k, v in sides.items()}
        far = "starboard" if (x1 + x2) / 2 >= nadir_x else "port"
        side = max(contrast, key=contrast.get)
        if contrast[far] > contrast[side] - 0.3:
            side = far
        shadow_contrast = contrast[side]
        _, dx, dy = sides[side]
        # shadow length: consecutive dark rows/cols walking away from the target
        run, skipped = 0, 0
        if dx:
            steps = range(x2, W) if dx > 0 else range(x1 - 1, -1, -1)
            line_mean = lambda c: float(gray[y1:y2, c].mean())
        else:
            steps = range(y2, H) if dy > 0 else range(y1 - 1, -1, -1)
            line_mean = lambda r: float(gray[r, x1:x2].mean())
        for c in steps:
            if line_mean(c) < bg_med - 0.5 * bg_std:
                run += 1
            elif run == 0 and skipped < 4:   # tolerate a few bright edge lines
                skipped += 1
            else:
                break
        shadow_m = run * mpp
        horiz = abs((x1 + x2) / 2 - nadir_x) * mpp + max(bw, 1) * mpp / 2
        slant = math.hypot(horiz, altitude) + shadow_m
        height = shadow_m * altitude / slant if shadow_m > 0 and shadow_contrast > 0.6 else None

        ev = [
            {"name": "Non-natural shape", "detail": f"{straight * 100:.0f}% of edges on straight segments",
             "score": round(straight, 2), "present": straight > 0.25},
            {"name": "Strong / structured return", "detail": f"target strength {strength:.1f}σ above seabed, edge density {edge_density:.2f}",
             "score": round(min(1, max(0, strength / 4)), 2), "present": strength > 1.5},
            {"name": "Target–shadow consistency", "detail": (f"shadow on {side} side{' (far range)' if side == far else ''}, "
                                                               f"contrast {shadow_contrast:.1f}σ, length ≈{shadow_m:.1f} m"),
             "score": round(min(1, max(0, shadow_contrast / 2)), 2), "present": shadow_contrast > 0.6},
        ]
        return ev, (round(height, 2) if height else None)

    # ------------------------------------------------------------------ Geolocation
    @staticmethod
    def geolocate(u, v, W, H, nav):
        mpp_x = nav["swath_m"] / W
        mpp_y = nav.get("along_m", nav["swath_m"] * H / W) / H
        dx = (u - W / 2) * mpp_x                  # +starboard
        dy = (H / 2 - v) * mpp_y                  # +forward (along-track, top of image = ahead)
        th = math.radians(nav["heading"])
        east = dx * math.cos(th) + dy * math.sin(th)
        north = -dx * math.sin(th) + dy * math.cos(th)
        lat = nav["lat"] + north / 111320.0
        lon = nav["lon"] + east / (111320.0 * math.cos(math.radians(nav["lat"])))
        return lat, lon, mpp_x, mpp_y

    # ------------------------------------------------------------------ full pipeline
    def analyze(self, img_bgr, nav, conf_thresh=0.3):
        x, enhanced = preprocess_sonar(img_bgr, IMG_SIZE)
        H, W = enhanced.shape
        sx, sy = W / IMG_SIZE, H / IMG_SIZE
        dets, anom, act, hm = self._infer(x)
        anom_full = cv2.resize(anom, (W, H), interpolation=cv2.INTER_CUBIC)
        nadir_x = W / 2

        results = []
        for x1, y1, x2, y2, score, c in dets:
            if score < conf_thresh:
                continue
            b = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            cname = CLASS_NAMES[int(c)]
            info = CLASS_INFO[cname]
            reg = anom_full[int(b[1]):max(int(b[1]) + 1, int(b[3])), int(b[0]):max(int(b[0]) + 1, int(b[2]))]
            a_score = float(np.percentile(reg, 90)) if reg.size else 0.0
            lat, lon, mx, my = self.geolocate((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, W, H, nav)
            ev, height = self._evidence(enhanced, b, nadir_x, mx, nav.get("altitude", 10.0))
            fp = info["man_made"] is False or (score < 0.4 and a_score < 0.3)
            results.append({
                "kind": "known_target", "class": cname, "target": info["label"], "category": info["category"],
                "confidence": round(float(score), 3), "anomaly_score": round(a_score, 2),
                "box": [round(v, 1) for v in b], "lat": round(lat, 6), "lon": round(lon, 6),
                "size_m": [round((b[2] - b[0]) * mx, 1), round((b[3] - b[1]) * my, 1)], "height_m": height,
                "evidence": ev,
                "status": "Probable False Positive" if fp else "Needs Verification",
                "fp_reason": ("biological target (fish) - not debris" if info["man_made"] is False else
                              "low confidence and seabed-like features" if fp else None),
            })

        # Unknown anomalies: strongly anomalous regions the known-class heads did not explain
        mask = (anom > 0.55).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        unknowns = []
        for i in range(1, n):
            gx, gy, gw, gh, area = stats[i]
            if area < 3:
                continue
            g = anom.shape[0]
            b = [float(gx / g * W), float(gy / g * H), float((gx + gw) / g * W), float((gy + gh) / g * H)]
            if any(iou(b, r["box"]) > 0.1 or (r["box"][0] <= (b[0] + b[2]) / 2 <= r["box"][2] and r["box"][1] <= (b[1] + b[3]) / 2 <= r["box"][3]) for r in results):
                continue
            a_score = float(anom[lab == i].max())
            unknowns.append((a_score, b))
        for a_score, b in sorted(unknowns, key=lambda t: -t[0])[:3]:
            lat, lon, mx, my = self.geolocate((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, W, H, nav)
            ev, height = self._evidence(enhanced, b, nadir_x, mx, nav.get("altitude", 10.0))
            results.append({
                "kind": "unknown_anomaly", "class": "unknown", "target": "UNKNOWN ANOMALY", "category": "Anomaly (no known class)",
                "confidence": None, "anomaly_score": round(a_score, 2), "box": [round(v, 1) for v in b],
                "lat": round(lat, 6), "lon": round(lon, 6),
                "size_m": [round((b[2] - b[0]) * mx, 1), round((b[3] - b[1]) * my, 1)], "height_m": height,
                "evidence": ev, "status": "Needs Verification", "fp_reason": None,
            })

        # visual products
        annotated = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        lw = max(2, W // 250)
        for k, r in enumerate(results):
            col = (0, 255, 0) if r["kind"] == "known_target" and r["status"] != "Probable False Positive" else \
                  (0, 200, 255) if r["kind"] == "unknown_anomaly" else (160, 160, 160)
            x1, y1, x2, y2 = map(int, r["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, lw)
            txt = f"#{k + 1} {r['class']} {r['confidence'] * 100:.0f}%" if r["confidence"] is not None else f"#{k + 1} anomaly {r['anomaly_score']:.2f}"
            fs = max(0.5, W / 900)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            ty = y1 - 6 if y1 - th - 8 > 0 else y2 + th + 6
            cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(annotated, txt, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
            crop = enhanced[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
            if crop.size:
                r["crop"] = _b64(cv2.resize(crop, (160, max(40, int(160 * crop.shape[0] / max(crop.shape[1], 1))))))

        known_hm = hm.max(0)
        return {
            "image_size": [W, H],
            "detections": results,
            "images": {
                "input": _b64(img_bgr),
                "preprocessed": _b64(enhanced),
                "activation": _b64(_overlay(enhanced, act)),
                "detection_heatmap": _b64(_overlay(enhanced, known_hm)),
                "anomaly": _b64(_overlay(enhanced, anom)),
                "result": _b64(annotated),
            },
            "global_anomaly": round(float(np.percentile(anom, 99)), 2),
        }
