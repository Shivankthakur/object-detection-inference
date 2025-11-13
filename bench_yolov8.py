import argparse, time, json, os, sys, statistics
from pathlib import Path

import cv2
import numpy as np
import psutil
import pandas as pd

# GPU stats
try:
    import pynvml
    NVML_OK = True
except Exception:
    NVML_OK = False

from ultralytics import YOLO

def percentiles(xs, ps=(50, 90, 99)):
    if not xs:
        return {f"p{p}": None for p in ps}
    xs_sorted = sorted(xs)
    out = {}
    for p in ps:
        k = (len(xs_sorted)-1) * (p/100)
        f = int(np.floor(k))
        c = int(np.ceil(k))
        if f == c:
            out[f"p{p}"] = xs_sorted[f]
        else:
            out[f"p{p}"] = xs_sorted[f] + (xs_sorted[c]-xs_sorted[f])*(k-f)
    return out

def init_nvml(gpu_index):
    if not NVML_OK:
        return None
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

def read_gpu_stats(handle):
    if handle is None:
        return {"gpu_util": None, "vram_used_mb": None, "power_w": None}
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    except pynvml.NVMLError:
        power_mw = None
    return {
        "gpu_util": util.gpu,
        "vram_used_mb": mem.used / (1024**2),
        "power_w": None if power_mw is None else power_mw / 1000.0
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov8m.pt", help="YOLOv8 model file (.pt or .engine)")
    ap.add_argument("--backend", type=str, default="torch", choices=["torch","tensorrt"], help="Torch or TensorRT engine")
    ap.add_argument("--device", type=str, default="cpu", help="'cpu' or '0' for first GPU")
    ap.add_argument("--source", type=str, required=True, help="Path to video file")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=500, help="Max frames to process (<= video length)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fp16", action="store_true", help="Use half precision (GPU only)")
    ap.add_argument("--save_csv", type=str, default="per_frame_metrics.csv")
    ap.add_argument("--save_json", type=str, default="summary.json")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads for CPU; 0=leave default")
    args = ap.parse_args()

    # Load model
    model = YOLO(args.model)

    if args.device == "cpu" and args.fp16:
        print("[WARN] --fp16 ignored on CPU.", file=sys.stderr)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.source}", file=sys.stderr)
        sys.exit(1)

    # Prepare NVML if GPU
    gpu_index = None
    nvml_handle = None
    if args.device != "cpu":
        try:
            gpu_index = int(args.device.split(":")[0]) if ":" in args.device else int(args.device)
        except Exception:
            gpu_index = 0
        nvml_handle = init_nvml(gpu_index)

    # Prime psutil CPU measurement
    psutil.cpu_percent(None)

    # Optional: control CPU threads for fair CPU runs
    if args.device == "cpu" and args.threads > 0:
        try:
            import torch
            torch.set_num_threads(args.threads)
            print(f"[INFO] torch.set_num_threads({args.threads})")
        except Exception:
            print("[WARN] Could not set torch threads.", file=sys.stderr)

    # Warmup
    warm = min(args.warmup, args.frames)
    ok, f0 = cap.read()
    if not ok:
        print("ERROR: video has no frames.", file=sys.stderr)
        sys.exit(1)
    for _ in range(warm):
        _ = model.predict(
            f0,
            imgsz=args.imgsz,
            device=args.device,
            half=(args.fp16 and args.device != "cpu"),
            verbose=False
        )

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    e2e_ms = []
    pp_ms, inf_ms, post_ms = [], [], []
    cpu_util, ram_mb = [], []
    gpu_util, vram_mb, power_w = [], [], []

    rows = []
    n = 0
    start_time = time.perf_counter()

    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            device=args.device,
            half=(args.fp16 and args.device != "cpu"),
            verbose=False
        )
        t1 = time.perf_counter()

        # Ultralytics returns per-image speed breakdown (ms)
        sp = results[0].speed  # dict: preprocess / inference / postprocess
        pp = float(sp.get("preprocess", np.nan))
        infer = float(sp.get("inference", np.nan))
        post = float(sp.get("postprocess", np.nan))
        e2e = (t1 - t0) * 1000.0

        # System stats
        cpu = psutil.cpu_percent(None)  # %
        mem = psutil.virtual_memory().used / (1024**2)

        gpu_s = read_gpu_stats(nvml_handle)

        # Accumulate
        e2e_ms.append(e2e)
        if not np.isnan(pp): pp_ms.append(pp)
        if not np.isnan(infer): inf_ms.append(infer)
        if not np.isnan(post): post_ms.append(post)
        cpu_util.append(cpu)
        ram_mb.append(mem)
        gpu_util.append(gpu_s["gpu_util"])
        vram_mb.append(gpu_s["vram_used_mb"])
        power_w.append(gpu_s["power_w"])

        rows.append({
            "frame": n,
            "e2e_ms": e2e,
            "pre_ms": pp,
            "infer_ms": infer,
            "post_ms": post,
            "cpu_util_pct": cpu,
            "ram_used_mb": mem,
            "gpu_util_pct": gpu_s["gpu_util"],
            "vram_used_mb": gpu_s["vram_used_mb"],
            "power_w": gpu_s["power_w"]
        })

        n += 1

    total_s = time.perf_counter() - start_time
    fps = n / total_s if total_s > 0 else None

    # Summaries
    e2e_stats = {
        "mean_ms": np.mean(e2e_ms) if e2e_ms else None,
        "stdev_ms": np.std(e2e_ms) if e2e_ms else None,
        **percentiles(e2e_ms)
    }
    pp_stats = {"mean_ms": np.mean(pp_ms) if pp_ms else None}
    inf_stats = {"mean_ms": np.mean(inf_ms) if inf_ms else None}
    post_stats = {"mean_ms": np.mean(post_ms) if post_ms else None}

    sys_stats = {
        "cpu_util_mean_pct": np.mean(cpu_util) if cpu_util else None,
        "ram_used_mean_mb": np.mean(ram_mb) if ram_mb else None,
        "gpu_util_mean_pct": np.nanmean(gpu_util) if gpu_util and any(x is not None for x in gpu_util) else None,
        "vram_used_mean_mb": np.nanmean(vram_mb) if vram_mb and any(x is not None for x in vram_mb) else None,
        "power_mean_w": np.nanmean(power_w) if power_w and any(x is not None for x in power_w) else None,
    }
    energy_per_frame_j = None
    if sys_stats["power_mean_w"] is not None and fps and fps > 0:
        energy_per_frame_j = sys_stats["power_mean_w"] / fps

    summary = {
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "imgsz": args.imgsz,
        "frames_run": n,
        "fps": fps,
        "latency_e2e_ms": e2e_stats,
        "latency_breakdown_mean_ms": {"pre": pp_stats["mean_ms"], "infer": inf_stats["mean_ms"], "post": post_stats["mean_ms"]},
        "system_means": sys_stats,
        "energy_per_frame_j": energy_per_frame_j,
        "duration_s": total_s
    }

    # Save artifacts
    pd.DataFrame(rows).to_csv(args.save_csv, index=False)
    with open(args.save_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
