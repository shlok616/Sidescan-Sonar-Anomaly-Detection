"""
Side-scan sonar dataset + preprocessing.

Reads the Roboflow "sonar_detect" export (YOLO txt labels; boxes or polygons) in ../dtst2
and builds CenterNet targets. The same `preprocess_sonar` is used at train and inference
time so the model sees identical inputs:
  colour palette -> single-channel intensity (device/palette independent)
  -> median filter (speckle) -> CLAHE (gain / TVG normalisation) -> resize.
"""
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 320
STRIDE = 8


def preprocess_sonar(img_bgr, size=IMG_SIZE):
    """Return (normalised float32 HxW in [0,1], uint8 enhanced image at original resolution)."""
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    den = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    small = cv2.resize(enh, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0, enh


def read_labels(path):
    """YOLO labels -> list of (cls, x1, y1, x2, y2) normalised. Polygons are boxed."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path).read().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, v = int(p[0]), list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append((c, max(0, x1), max(0, y1), min(1, x2), min(1, y2)))
    return out


def gaussian_radius(h, w, min_overlap=0.7):
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(hm, cx, cy, r):
    d = 2 * r + 1
    sigma = d / 6
    ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    H, W = hm.shape
    l, ri = min(cx, r), min(W - cx, r + 1)
    t, bo = min(cy, r), min(H - cy, r + 1)
    np.maximum(hm[cy - t:cy + bo, cx - l:cx + ri], g[r - t:r + bo, r - l:r + ri], out=hm[cy - t:cy + bo, cx - l:cx + ri])


class SonarDetectionDataset(Dataset):
    def __init__(self, root, split="train", augment=False, num_classes=4, max_objs=64):
        self.images = sorted(glob.glob(os.path.join(root, split, "images", "*.jpg")))
        self.root, self.split = root, split
        self.augment, self.K, self.max_objs = augment, num_classes, max_objs
        self.cache = {}

    def __len__(self):
        return len(self.images)

    def label_path(self, img_path):
        return os.path.join(self.root, self.split, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")

    def load(self, i):
        if i not in self.cache:
            img = cv2.imread(self.images[i])
            x, _ = preprocess_sonar(img, IMG_SIZE)
            self.cache[i] = (x, read_labels(self.label_path(self.images[i])))
        return self.cache[i]

    def _augment(self, x, boxes):
        boxes = [list(b) for b in boxes]
        if random.random() < 0.5:   # port/starboard mirror
            x = x[:, ::-1]
            boxes = [[c, 1 - x2, y1, 1 - x1, y2] for c, x1, y1, x2, y2 in boxes]
        if random.random() < 0.5:   # along-track flip
            x = x[::-1, :]
            boxes = [[c, x1, 1 - y2, x2, 1 - y1] for c, x1, y1, x2, y2 in boxes]
        if random.random() < 0.3:   # 90 deg rotation (sonar is oriented arbitrarily in this dataset)
            x = np.rot90(x)  # counter-clockwise: (x,y)->(y,1-x)
            boxes = [[c, y1, 1 - x2, y2, 1 - x1] for c, x1, y1, x2, y2 in boxes]
        if random.random() < 0.6:   # random crop / zoom
            s = random.uniform(0.6, 1.0)
            ox, oy = random.uniform(0, 1 - s), random.uniform(0, 1 - s)
            n = []
            for c, x1, y1, x2, y2 in boxes:
                a = [(x1 - ox) / s, (y1 - oy) / s, (x2 - ox) / s, (y2 - oy) / s]
                b = [min(max(v, 0), 1) for v in a]
                area0 = (a[2] - a[0]) * (a[3] - a[1])
                if area0 > 0 and (b[2] - b[0]) * (b[3] - b[1]) / area0 > 0.4:
                    n.append([c] + b)
            if n or not boxes:
                H = x.shape[0]
                x = cv2.resize(np.ascontiguousarray(x[int(oy * H):int((oy + s) * H), int(ox * H):int((ox + s) * H)]), (H, H))
                boxes = n
        x = np.ascontiguousarray(x)
        # gain / contrast / gamma / speckle noise (domain variation between sonar devices)
        x = np.clip(x * random.uniform(0.7, 1.3) + random.uniform(-0.15, 0.15), 0, 1)
        x = x ** random.uniform(0.7, 1.4)
        if random.random() < 0.5:
            x = x * np.random.gamma(8, 1 / 8, x.shape).astype(np.float32)
        x = np.clip(x + np.random.normal(0, random.uniform(0, 0.05), x.shape), 0, 1).astype(np.float32)
        return x, boxes

    def __getitem__(self, i):
        x, boxes = self.load(i)
        if self.augment:
            x, boxes = self._augment(x, boxes)
        S = IMG_SIZE // STRIDE
        hm = np.zeros((self.K, S, S), np.float32)
        wh = np.zeros((self.max_objs, 2), np.float32)
        off = np.zeros((self.max_objs, 2), np.float32)
        ind = np.zeros(self.max_objs, np.int64)
        msk = np.zeros(self.max_objs, np.float32)
        for k, (c, x1, y1, x2, y2) in enumerate(boxes[: self.max_objs]):
            w, h = (x2 - x1) * S, (y2 - y1) * S
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x1 + x2) / 2 * S, (y1 + y2) / 2 * S
            ix, iy = min(int(cx), S - 1), min(int(cy), S - 1)
            draw_gaussian(hm[c], ix, iy, max(0, gaussian_radius(math.ceil(h), math.ceil(w))))
            wh[k] = np.log([max(w, 0.25), max(h, 0.25)])
            off[k] = [cx - ix, cy - iy]
            ind[k] = iy * S + ix
            msk[k] = 1
        gt = np.zeros((self.max_objs, 6), np.float32)
        for k, (c, x1, y1, x2, y2) in enumerate(boxes[: self.max_objs]):
            gt[k] = [x1 * IMG_SIZE, y1 * IMG_SIZE, x2 * IMG_SIZE, y2 * IMG_SIZE, c, 1]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x))[None],
            "hm": torch.from_numpy(hm), "wh": torch.from_numpy(wh), "off": torch.from_numpy(off),
            "ind": torch.from_numpy(ind), "mask": torch.from_numpy(msk), "gt": torch.from_numpy(gt),
        }
