import argparse, time, sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def parse_classes(arg, names):
    if arg is None or arg.strip() == "":
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    idxs = []
    for it in items:
        if it.isdigit():
            idxs.append(int(it))
        else:
            found = [i for i, n in (names or {}).items() if str(n).lower() == it.lower()]
            if not found:
                print(f"[WARN] class '{it}' not found in model.names; skipping", file=sys.stderr)
            else:
                idxs.append(found[0])
    idxs = sorted(set([i for i in idxs if i is not None]))
    return idxs if idxs else None

def quote_for_concat(path: Path) -> str:
    p = path.resolve().as_posix().replace("'", r"'\''")
    return f"file '{p}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8m.pt", help="Ultralytics model (.pt or .engine)")
    ap.add_argument("--source", required=True, help="Video path or webcam index like '0'")
    ap.add_argument("--device", default="cpu", help="'cpu' or GPU index like '0'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=str, default=None, help="e.g., 'person,car' or '0,2'")
    ap.add_argument("--fp16", action="store_true", help="Half precision (GPU only)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--show", action="store_true", help="Live preview (not timed)")
    ap.add_argument("--line_width", type=int, default=2)
    ap.add_argument("--hide_conf", action="store_true")
    ap.add_argument("--stamp_wallclock", action="store_true", help="Overlay wall-clock t and per-frame dt")
    ap.add_argument("--out_dir", default="out_vfr", help="Directory to write frames/frames.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    concat_txt = out_dir / "frames.txt"

    # Load model
    model = YOLO(args.model)

    # Source (webcam or file)
    source = args.source
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Optional class filter
    class_filter = parse_classes(args.classes, getattr(model, "names", None))

    # Warmup (improves stability, not part of timing)
    ok, warm = cap.read()
    if not ok:
        print("ERROR: no frames available.", file=sys.stderr); sys.exit(1)
    _ = model.predict(warm, imgsz=args.imgsz, device=args.device,
                      half=(args.fp16 and args.device != "cpu"),
                      conf=args.conf, classes=class_filter, verbose=False)
    # reset to start for files
    if not isinstance(source, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process
    frame_i = 0
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else float("inf")
    wall_elapsed = 0.0  # for on-frame stamp

    # Open concat file
    ftxt = open(concat_txt, "w", encoding="utf-8")
    ftxt.write("ffconcat version 1.0\n")

    last_frame_path = None

    print(f"[INFO] Writing PNG frames to: {frames_dir}")
    print(f"[INFO] Timeline file: {concat_txt}")
    print("[INFO] Building per-frame durations from actual processing time (VFR).")

    while frame_i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        # Inference
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            device=args.device,
            half=(args.fp16 and args.device != "cpu"),
            conf=args.conf,
            classes=class_filter,
            verbose=False
        )
        res = results[0]

        # Draw
        annotated = res.plot(
            line_width=args.line_width,
            labels=True,
            conf=not args.hide_conf,
            boxes=True
        )

        t1 = time.perf_counter()
        dt = max(1e-3, t1 - t0)  # duration for this frame; clamp to 1ms minimum (FFmpeg requires >0)
        wall_elapsed += dt

        # Optional overlay
        if args.stamp_wallclock:
            txt = f"t={wall_elapsed:6.2f}s  dt={dt*1000.0:5.1f} ms"
            cv2.putText(annotated, txt, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(annotated, txt, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Save PNG
        frame_path = frames_dir / f"frame_{frame_i:06d}.png"
        cv2.imwrite(str(frame_path), annotated)

        # Write to concat file (file first, then duration)
        ftxt.write(quote_for_concat(frame_path) + "\n")
        ftxt.write(f"duration {dt:.6f}\n")


        # Live preview (not affecting dt)
        if args.show:
            cv2.imshow("YOLOv8 VFR Annotated", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        last_frame_path = frame_path
        frame_i += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    # FFmpeg concat demuxer needs the last file repeated once (without a duration)
    if last_frame_path is not None:
        ftxt.write(quote_for_concat(last_frame_path) + "\n")

    ftxt.close()

    print(f"[DONE] Wrote {frame_i} frames and timeline to: {out_dir}")
    print("\nNext step (build VFR MP4 with FFmpeg):")
    print(f'ffmpeg -f concat -safe 0 -i "{concat_txt}" -c:v libx264 -pix_fmt yuv420p -vsync vfr "{out_dir / "out_vfr.mp4"}"')

if __name__ == "__main__":
    main()
