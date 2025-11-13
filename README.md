# Object Detection Inference Toolkit

A lightweight toolkit for benchmarking and visualizing Ultralytics YOLOv8 object detection models on video sources. The scripts included here let you compare CPU vs GPU performance, generate annotated videos, and build variable-frame-rate timelines that reflect real per-frame latency.

## Key scripts

- `bench_yolov8.py` — run a video through YOLOv8 and log latency, FPS, and system stats per frame, saving results to CSV/JSON.
- `annotate_yolov8_video.py` — render an annotated video with bounding boxes (fixed FPS output).
- `annotate_yolov8_vfr.py` — export per-frame PNGs and an FFmpeg concat file that preserves variable frame timing based on measured inference time.
- `compare_json.py` — merge two benchmark summaries (e.g., CPU vs GPU) into a CSV and quick console table.
- `viz_yolov8_compare_multilib.py` — build static plots or an interactive dashboard comparing CPU/GPU benchmark CSVs.
- `test_device.py` — inspect local CUDA and CPU availability.

Sample videos and benchmark outputs (e.g., `sample.mp4`, `cpu.csv`, `gpu.csv`) live under `data/` files in the repo root.

**Sample video source:** The `sample.mp4` file is sourced from the [Road Traffic Video Monitoring dataset](https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring?select=road_trafifc.mp4) on Kaggle.

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for GPU benchmarks or TensorRT runs.
- FFmpeg (optional) if you plan to convert the VFR frame export into a playable MP4.

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Benchmark YOLOv8 inference

```bash
python bench_yolov8.py --source data/sample.mp4 --model yolov8m.pt --device cpu
```

Key flags:

- `--device`: `cpu`, `0`, `1`, etc.
- `--backend`: `torch` (default) or `tensorrt` (requires an exported engine).
- `--imgsz`: image resolution (default 640).
- `--fp16`: enable half precision on GPU.
- `--frames`: maximum number of frames to process.

The script writes `per_frame_metrics.csv` and `summary.json` by default.

**Example commands used for current benchmark results:**

```bash
# CPU benchmark
python bench_yolov8.py --model yolov8m.pt --backend torch --device cpu --source data/sample.mp4 --imgsz 640 --frames 500 --warmup 20 --threads 8 --save_json data/cpu.json --save_csv data/cpu.csv

# GPU benchmark
python bench_yolov8.py --model yolov8m.pt --backend torch --device 0 --source data/sample.mp4 --imgsz 640 --frames 500 --warmup 20 --fp16 --save_json data/gpu.json --save_csv data/gpu.csv
```

### 2. Render an annotated video

```bash
python annotate_yolov8_video.py --source data/sample.mp4 --out out/annotated.mp4 --model yolov8m.pt --device cpu
```

Useful options:

- `--classes person,car` to restrict detections.
- `--conf 0.35` to adjust confidence threshold.
- `--show` to preview frames while writing.

**Live webcam annotation:**

```bash
python annotate_yolov8_video.py --model yolov8m.pt --source 0 --device 0 --imgsz 480 --fp16 --show --out webcam_out.mp4
```

### 3. Variable-frame-rate export

```bash
python annotate_yolov8_vfr.py --source data/sample.mp4 --out_dir out_vfr --stamp_wallclock
```

This produces PNG frames plus `frames.txt`. Assemble the final MP4 with FFmpeg:

```bash
ffmpeg -f concat -safe 0 -i "out_vfr/frames.txt" -c:v libx264 -pix_fmt yuv420p -vsync vfr out_vfr/out_vfr.mp4
```

### 4. Compare benchmark summaries

After running CPU and GPU benchmarks:

```bash
python compare_json.py
```

Ensure `cpu.json` and `gpu.json` exist in the working directory.

### 5. Visualize benchmark results

Generate Matplotlib, Seaborn, or Plotly comparisons:

```bash
python viz_yolov8_compare_multilib.py --cpu_csv cpu.csv --gpu_csv gpu.csv --cpu_json cpu.json --gpu_json gpu.json --out viz_out --lib all
```

Artifacts (PNGs and `dashboard.html`) are written to `viz_out/`.

## Tips

- Run `python test_device.py` to confirm CUDA visibility before launching GPU jobs.
- For reproducible results, pin the number of CPU threads with `--threads` in `bench_yolov8.py`.
- Large models may require updating `--imgsz` or enabling `--fp16` to fit in GPU memory.
