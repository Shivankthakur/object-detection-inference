import argparse, time, sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def parse_classes(arg, names):
    """
    Accepts either comma-separated ids (e.g., "0,2,5") or labels (e.g., "person,car,dog").
    Returns a sorted list of class indices or None.
    """
    if arg is None or arg.strip() == "":
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    idxs = []
    for it in items:
        if it.isdigit():
            idxs.append(int(it))
        else:
            # map label to id (case-insensitive)
            found = [i for i, n in (names or {}).items() if str(n).lower() == it.lower()]
            if not found:
                print(f"[WARN] class '{it}' not found in model.names; skipping", file=sys.stderr)
            else:
                idxs.append(found[0])
    idxs = sorted(set([i for i in idxs if i is not None]))
    return idxs if idxs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8m.pt", help="Ultralytics model (.pt or .engine)")
    ap.add_argument("--source", required=True, help="Video path or webcam index (e.g., 0)")
    ap.add_argument("--out", default="annotated.mp4", help="Output video path")
    ap.add_argument("--device", default="cpu", help="'cpu' or GPU index like '0'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--classes", type=str, default=None, help="Comma-separated class ids or names (e.g., '0,2' or 'person,car')")
    ap.add_argument("--fp16", action="store_true", help="Use half precision (GPU only)")
    ap.add_argument("--show", action="store_true", help="Show a live preview window")
    ap.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    ap.add_argument("--fourcc", type=str, default="mp4v", help="Video codec (e.g., mp4v, avc1, XVID)")
    ap.add_argument("--line_width", type=int, default=2, help="Box/label line thickness")
    ap.add_argument("--hide_conf", action="store_true", help="Hide confidence text on boxes")
    args = ap.parse_args()

    # Load model
    model = YOLO(args.model)

    # Source can be int (webcam) or path
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Video props
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 30.0  # fallback
    if w == 0 or h == 0:
        # Try to read a frame to infer size
        ok, fr = cap.read()
        if not ok:
            print("ERROR: no frames available.", file=sys.stderr); sys.exit(1)
        h, w = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Video writer
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    # Class filter
    class_filter = parse_classes(args.classes, getattr(model, "names", None))

    # Warmup (helps stabilize timing/first-frame JIT)
    ok, warm = cap.read()
    if ok:
        _ = model.predict(warm, imgsz=args.imgsz, device=args.device,
                          half=(args.fp16 and args.device != "cpu"),
                          conf=args.conf, classes=class_filter, verbose=False)

    # Main loop
    t_prev = time.perf_counter()
    ema_fps = None
    frame_i = 0
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else float("inf")

    print(f"[INFO] writing to: {out_path} ({w}x{h} @{fps:.2f} FPS)")
    if class_filter:
        print(f"[INFO] filtering classes: {class_filter}")

    while frame_i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
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

        # draw boxes/labels
        annotated = res.plot(
            line_width=args.line_width,
            labels=True,
            conf=not args.hide_conf,
            boxes=True
        )

        # FPS overlay (EMA for stability)
        t1 = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t1 - t_prev))
        ema_fps = inst_fps if ema_fps is None else (0.9*ema_fps + 0.1*inst_fps)
        t_prev = t1
        cv2.putText(annotated, f"FPS: {ema_fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {ema_fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        if args.show:
            cv2.imshow("YOLOv8 Annotated", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        frame_i += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"[DONE] wrote {frame_i} frames to {out_path}")

if __name__ == "__main__":
    main()
