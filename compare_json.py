import json, pandas as pd
def pick(p): return None if p is None else round(p,2)
cpu = json.load(open("cpu.json"))
gpu = json.load(open("gpu.json"))
rows=[]
for name,js in [("CPU",cpu),("GPU",gpu)]:
    lat = js["latency_e2e_ms"]
    brk = js["latency_breakdown_mean_ms"]
    sysm = js["system_means"]
    rows.append({
        "Run": name,
        "FPS": round(js["fps"],2),
        "e2e_mean_ms": pick(lat["mean_ms"]),
        "e2e_p50_ms": pick(lat["p50"]),
        "e2e_p90_ms": pick(lat["p90"]),
        "e2e_p99_ms": pick(lat["p99"]),
        "infer_mean_ms": pick(brk["infer"]),
        "pre_mean_ms": pick(brk["pre"]),
        "post_mean_ms": pick(brk["post"]),
        "CPU_util_%": pick(sysm["cpu_util_mean_pct"]),
        "GPU_util_%": pick(sysm["gpu_util_mean_pct"]),
        "VRAM_MB": pick(sysm["vram_used_mean_mb"]),
        "Power_W": pick(sysm["power_mean_w"]),
        "Energy_J_per_frame": pick(js["energy_per_frame_j"])
    })
df = pd.DataFrame(rows)
df.to_csv("benchmark_results_summary.csv", index=False)
print(df.to_string(index=False))
print("\nSpeedup (GPU/CPU FPS):", round(gpu["fps"]/cpu["fps"],2))
print("Latency reduction (CPU_mean / GPU_mean):", round(cpu["latency_e2e_ms"]["mean_ms"]/gpu["latency_e2e_ms"]["mean_ms"],2), "x")