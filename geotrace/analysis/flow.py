"""Ego-motion from low-res frames with numpy phase correlation.
 yaw: horizontal shift of the far-field band (sky/buildings); speed proxy: vertical shift of the road patch ahead."""
import json
import math
import os
import sys

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NF, W, H = 40, 160, 90
FPS = 8.0
HFOV = 100.0
DEGPX = HFOV / W


def hann(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


WIN_B = hann(28, 160)
WIN_R = hann(28, 80)


def pcorr(a, b, win):
    """shift d such that b(x) = a(x - d); subpixel; returns dx, dy, peak"""
    FA = np.fft.fft2(a * win)
    FB = np.fft.fft2(b * win)
    R = FB * np.conj(FA)
    R /= np.abs(R) + 1e-6
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    h, w = r.shape

    def sub(c, m, p):
        den = (m - 2 * c + p)
        return 0.0 if abs(den) < 1e-9 else 0.5 * (m - p) / den

    ry = sub(r[py, px], r[(py - 1) % h, px], r[(py + 1) % h, px])
    rx = sub(r[py, px], r[py, (px - 1) % w], r[py, (px + 1) % w])
    dy = py + ry
    dx = px + rx
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dx, dy, float(r[py, px])


def seg_features(F):
    F = F.astype(np.float32)
    F -= F.mean(axis=(1, 2), keepdims=True)
    yaw_px, dys, conf_r = [], [], []
    for k in range(F.shape[0] - 1):
        a, b = F[k], F[k + 1]
        dx, _, c1 = pcorr(a[8:36, :], b[8:36, :], WIN_B)
        _, dy, c2 = pcorr(a[46:74, 40:120], b[46:74, 40:120], WIN_R)
        yaw_px.append(dx if c1 > 0.03 else 0.0)
        dys.append(dy if c2 > 0.03 else 0.0)
        conf_r.append(c2)
    yaw_deg = -float(np.sum(yaw_px)) * DEGPX          # clockwise (right) positive
    return {"yaw_deg": yaw_deg, "v_px": float(np.median(np.maximum(dys, 0))) * FPS, "v_mean": float(np.mean(np.maximum(dys, 0))) * FPS}


def run_trip(trip):
    fp = os.path.join(D, f"frames_{trip}.bin")
    arr = np.memmap(fp, dtype=np.uint8, mode="r")
    n = arr.size // (NF * W * H)
    arr = arr.reshape(n, NF, H, W)
    return [seg_features(arr[i]) for i in range(n)]


if __name__ == "__main__":
    for trip in sys.argv[1:]:
        f = run_trip(trip)
        json.dump(f, open(os.path.join(D, f"flow_{trip}.json"), "w"))
        y = np.array([x["yaw_deg"] for x in f])
        v = np.array([x["v_px"] for x in f])
        print(trip, "segments", len(f), "|yaw|>30deg:", int((np.abs(y) > 30).sum()), "v_px percentiles 10/50/90:", np.percentile(v, [10, 50, 90]).round(2))
