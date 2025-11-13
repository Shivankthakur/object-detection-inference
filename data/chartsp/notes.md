# Check output parity

> python compare_cpu_gpu_outputs.py --model yolov8m.pt --source data/sample.mp4 --imgsz 640 --frames 200


# Run CPU benchmark

python bench_yolov8.py ^
  --model yolov8m.pt ^
  --backend torch ^
  --device cpu ^
  --source data/sample.mp4 ^
  --imgsz 640 ^
  --frames 500 ^
  --warmup 20 ^
  --threads 8 ^
  --save_json cpu.json ^
  --save_csv cpu.csv


# Run GPU benchmark

python bench_yolov8.py ^
  --model yolov8m.pt ^
  --backend torch ^
  --device 0 ^
  --source data/sample.mp4 ^
  --imgsz 640 ^
  --frames 500 ^
  --warmup 20 ^
  --fp16 ^
  --save_json gpu.json ^
  --save_csv gpu.csv


# Annotate video

python annotate_yolov8_video.py ^
  --model yolov8m.pt ^
  --source data/sample.mp4 ^
  --device 0 ^
  --imgsz 480 ^
  --fp16 ^
  --show ^
  --out webcam_out.mp4

python annotate_yolov8_video.py ^
  --model yolov8m.pt ^
  --source data/sample.mp4 ^
  --device cpu ^
  --imgsz 480 ^
  --conf 0.25 ^
  --show ^
  --out out_cpu_webcam.mp4

# webcam
python annotate_yolov8_video.py ^
  --model yolov8m.pt ^
  --source 0 ^
  --device 0 ^
  --imgsz 480 ^
  --fp16 ^
  --show ^
  --out webcam_out.mp4