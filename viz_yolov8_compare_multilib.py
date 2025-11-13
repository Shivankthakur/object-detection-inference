import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Backends load on demand
import matplotlib.pyplot as plt

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def read_runs(cpu_csv, gpu_csv, cpu_json=None, gpu_json=None):
    cpu = pd.read_csv(cpu_csv)
    gpu = pd.read_csv(gpu_csv)
    js_cpu = json.load(open(cpu_json)) if cpu_json else None
    js_gpu = json.load(open(gpu_json)) if gpu_json else None
    return cpu, gpu, js_cpu, js_gpu

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# ---------- Shared computations ----------
def ecdf_xy(x):
    xs = np.sort(np.asarray(x))
    ys = np.arange(1, len(xs)+1)/len(xs) if len(xs) else np.array([])
    return xs, ys

def stage_means(js):
    if not js: return None
    bd = js.get("latency_breakdown_mean_ms", {})
    if not bd: return None
    return float(bd.get("pre") or np.nan), float(bd.get("infer") or np.nan), float(bd.get("post") or np.nan)

def mean_or_nan(s):
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else np.nan

# ---------- Matplotlib (your original + fix + extras) ----------
def mpl_plots(cpu, gpu, js_cpu, js_gpu, out):
    # 1) Speedup
    if js_cpu and js_gpu:
        speedup = js_gpu["fps"] / js_cpu["fps"]
        plt.figure()
        plt.bar(["GPU/CPU"], [speedup])
        plt.title("Overall speedup (FPS ratio)"); plt.ylabel("×")
        plt.text(0, speedup, f"{speedup:.2f}×", ha="center", va="bottom")
        savefig(f"{out}/01_speedup.png")

    # 2) Mean latency bar
    means = [cpu["e2e_ms"].mean(), gpu["e2e_ms"].mean()]
    plt.figure()
    plt.bar(["CPU","GPU"], means)
    plt.title("Mean end-to-end latency (ms)"); plt.ylabel("ms")
    for i,v in enumerate(means): plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    savefig(f"{out}/02_mean_latency.png")

    # 3) Stage breakdown stacked
    cm = stage_means(js_cpu); gm = stage_means(js_gpu)
    if cm and gm:
        pre = [cm[0], gm[0]]; inf = [cm[1], gm[1]]; post = [cm[2], gm[2]]
        x = np.arange(2)
        plt.figure()
        p1 = plt.bar(x, pre)
        p2 = plt.bar(x, inf, bottom=pre)
        p3 = plt.bar(x, post, bottom=np.array(pre)+np.array(inf))
        plt.xticks(x, ["CPU","GPU"]); plt.ylabel("ms"); plt.title("Latency breakdown (mean ms)")
        plt.legend((p1[0], p2[0], p3[0]), ("preprocess","inference","postprocess"))
        savefig(f"{out}/03_breakdown_stacked.png")

    # 4) Boxplot (fixed deprecation: use tick_labels)
    plt.figure()
    plt.boxplot([cpu["e2e_ms"].values, gpu["e2e_ms"].values], tick_labels=["CPU","GPU"], showfliers=True)
    plt.ylabel("ms"); plt.title("End-to-end latency (boxplot)")
    savefig(f"{out}/04_box_latency.png")

    # 5) Histogram overlay
    plt.figure()
    plt.hist(cpu["e2e_ms"], bins=40, alpha=0.6, density=True, label="CPU")
    plt.hist(gpu["e2e_ms"], bins=40, alpha=0.6, density=True, label="GPU")
    plt.xlabel("ms"); plt.ylabel("density"); plt.title("Latency distribution (hist)"); plt.legend()
    savefig(f"{out}/05_hist_latency.png")

    # 6) ECDF
    xc,yc = ecdf_xy(cpu["e2e_ms"]); xg,yg = ecdf_xy(gpu["e2e_ms"])
    plt.figure()
    if len(xc): plt.plot(xc,yc,label="CPU")
    if len(xg): plt.plot(xg,yg,label="GPU")
    plt.xlabel("ms"); plt.ylabel("fraction ≤ ms"); plt.title("Latency ECDF"); plt.grid(True, alpha=0.3); plt.legend()
    savefig(f"{out}/06_ecdf_latency.png")

    # 7) Percentile curve
    P = np.linspace(0,100,101)
    plt.figure()
    plt.plot(P, np.percentile(cpu["e2e_ms"], P), label="CPU")
    plt.plot(P, np.percentile(gpu["e2e_ms"], P), label="GPU")
    plt.xlabel("percentile"); plt.ylabel("ms"); plt.title("Latency percentile curve"); plt.grid(True, alpha=0.3); plt.legend()
    savefig(f"{out}/07_percentiles.png")

    # 8) Time series
    plt.figure()
    plt.plot(cpu.index, cpu["e2e_ms"], label="CPU")
    plt.plot(gpu.index, gpu["e2e_ms"], label="GPU")
    plt.xlabel("frame"); plt.ylabel("ms"); plt.title("Per-frame latency over time"); plt.legend()
    savefig(f"{out}/08_time_series.png")

    # 9) Scatter util
    plt.figure()
    plotted=False
    if "cpu_util_pct" in cpu.columns:
        plt.scatter(cpu["cpu_util_pct"], cpu["e2e_ms"], s=8, label="CPU"); plotted=True
    if "gpu_util_pct" in gpu.columns:
        plt.scatter(gpu["gpu_util_pct"], gpu["e2e_ms"], s=8, label="GPU"); plotted=True
    if plotted:
        plt.xlabel("utilization (%)"); plt.ylabel("latency (ms)"); plt.title("Latency vs utilization"); plt.legend()
        savefig(f"{out}/09_scatter_util.png")
    else:
        plt.close()

    # 10/11) Power & energy bars
    if js_cpu and js_gpu:
        power = [js_cpu["system_means"].get("power_mean_w"), js_gpu["system_means"].get("power_mean_w")]
        energy = [js_cpu.get("energy_per_frame_j"), js_gpu.get("energy_per_frame_j")]

        if any(x is not None for x in power):
            plt.figure()
            vals = [x if x is not None else 0.0 for x in power]
            plt.bar(["CPU","GPU"], vals)
            for i,v in enumerate(power):
                txt = "N/A" if v is None else f"{v:.1f}"
                plt.text(i, vals[i], txt, ha="center", va="bottom")
            plt.ylabel("W"); plt.title("Mean power (lower is better)")
            savefig(f"{out}/10_power.png")

        if any(x is not None for x in energy):
            plt.figure()
            vals = [x if x is not None else 0.0 for x in energy]
            plt.bar(["CPU","GPU"], vals)
            for i,v in enumerate(energy):
                txt = "N/A" if v is None else f"{v:.3f}"
                plt.text(i, vals[i], txt, ha="center", va="bottom")
            plt.ylabel("J/frame"); plt.title("Energy per frame (lower is better)")
            savefig(f"{out}/11_energy.png")

    # 12) Violin (matplotlib)
    plt.figure()
    plt.violinplot([cpu["e2e_ms"].values, gpu["e2e_ms"].values], showmeans=True, showmedians=True, showextrema=True)
    plt.xticks([1,2], ["CPU","GPU"])
    plt.ylabel("ms"); plt.title("Latency (violin)")
    savefig(f"{out}/12_violin_latency.png")

    # 16) Rolling FPS
    for window in [30]:
        cpu_fps = 1000.0 / cpu["e2e_ms"]; gpu_fps = 1000.0 / gpu["e2e_ms"]
        plt.figure()
        plt.plot(cpu_fps.rolling(window, min_periods=1).mean(), label=f"CPU (roll {window})")
        plt.plot(gpu_fps.rolling(window, min_periods=1).mean(), label=f"GPU (roll {window})")
        plt.axhline(cpu_fps.mean(), linestyle="--", label=f"CPU mean {cpu_fps.mean():.1f}")
        plt.axhline(gpu_fps.mean(), linestyle="--", label=f"GPU mean {gpu_fps.mean():.1f}")
        plt.xlabel("frame"); plt.ylabel("FPS"); plt.title("Rolling FPS (sustained throughput)"); plt.legend()
        savefig(f"{out}/16_rolling_fps.png")

    # 17) Frames vs time
    t_cpu = np.cumsum(cpu["e2e_ms"].values)/1000.0
    t_gpu = np.cumsum(gpu["e2e_ms"].values)/1000.0
    plt.figure()
    plt.plot(t_cpu, np.arange(1,len(t_cpu)+1), label="CPU")
    plt.plot(t_gpu, np.arange(1,len(t_gpu)+1), label="GPU")
    plt.xlabel("time (s)"); plt.ylabel("frames processed"); plt.title("Frames vs time"); plt.legend()
    savefig(f"{out}/17_frames_vs_time.png")


# ---------- Seaborn (pretty static plots) ----------
def seaborn_plots(cpu, gpu, js_cpu, js_gpu, out):
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    # Combine runs
    cpu2 = cpu.copy(); cpu2["device"]="CPU"
    gpu2 = gpu.copy(); gpu2["device"]="GPU"
    df = pd.concat([cpu2, gpu2], ignore_index=True)

    # Mean latency bar with CI
    plt.figure()
    sns.barplot(data=df, x="device", y="e2e_ms", estimator=np.mean, errorbar=("ci",95))
    plt.title("Mean end-to-end latency (95% CI)"); plt.ylabel("ms")
    savefig(f"{out}/sbn_01_mean_latency.png")

    # Violin
    plt.figure()
    sns.violinplot(data=df, x="device", y="e2e_ms", inner="quartile", cut=0)
    plt.title("Latency distribution (violin)"); plt.ylabel("ms")
    savefig(f"{out}/sbn_02_violin.png")

    # ECDF
    plt.figure()
    sns.ecdfplot(data=df, x="e2e_ms", hue="device")
    plt.xlabel("ms"); plt.ylabel("fraction ≤ ms"); plt.title("Latency ECDF")
    savefig(f"{out}/sbn_03_ecdf.png")

    # Histogram (stacked-ish via multiple)
    plt.figure()
    sns.histplot(data=df, x="e2e_ms", hue="device", stat="density", bins=40, common_norm=False, element="step")
    plt.title("Latency histogram"); plt.xlabel("ms")
    savefig(f"{out}/sbn_04_hist.png")

    # Utilization → Latency (regression)
    if "cpu_util_pct" in cpu.columns:
        plt.figure()
        sns.regplot(data=cpu, x="cpu_util_pct", y="e2e_ms", scatter_kws={"s":8}, line_kws={"lw":2})
        plt.title("CPU: utilization vs latency")
        savefig(f"{out}/sbn_05_cpu_util_reg.png")
    if "gpu_util_pct" in gpu.columns:
        plt.figure()
        sns.regplot(data=gpu, x="gpu_util_pct", y="e2e_ms", scatter_kws={"s":8}, line_kws={"lw":2})
        plt.title("GPU: utilization vs latency")
        savefig(f"{out}/sbn_06_gpu_util_reg.png")

    # Correlation heatmaps
    for name,df0 in [("CPU",cpu),("GPU",gpu)]:
        cols = [c for c in ["e2e_ms","pre_ms","infer_ms","post_ms","cpu_util_pct","gpu_util_pct","ram_used_mb","vram_used_mb","power_w"] if c in df0.columns]
        if len(cols)>=2:
            corr = df0[cols].corr()
            plt.figure()
            ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, cbar=True)
            plt.title(f"{name} correlation heatmap")
            savefig(f"{out}/sbn_07_corr_{name.lower()}.png")

    # Stage breakdown stacked (mean)
    cm = stage_means(js_cpu); gm = stage_means(js_gpu)
    if cm and gm:
        dd = pd.DataFrame({
            "device":["CPU","GPU"],
            "pre":[cm[0],gm[0]],
            "infer":[cm[1],gm[1]],
            "post":[cm[2],gm[2]],
        })
        ddm = dd.melt(id_vars="device", var_name="stage", value_name="ms")
        plt.figure()
        sns.barplot(data=ddm, x="device", y="ms", hue="stage", estimator=np.sum)
        plt.title("Latency breakdown (mean ms)")
        savefig(f"{out}/sbn_08_breakdown.png")


# ---------- Plotly (interactive dashboard) ----------
def plotly_dashboard(cpu, gpu, js_cpu, js_gpu, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    cpu_fps = 1000.0/ cpu["e2e_ms"]; gpu_fps = 1000.0/ gpu["e2e_ms"]
    cm = stage_means(js_cpu); gm = stage_means(js_gpu)

    rows, cols = 3, 2
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=("Mean latency","Stage breakdown",
                                        "ECDF","Per-frame latency",
                                        "Rolling FPS","Power/Energy"))

    # Mean latency
    fig.add_trace(go.Bar(x=["CPU","GPU"], y=[mean_or_nan(cpu["e2e_ms"]), mean_or_nan(gpu["e2e_ms"])]),
                  row=1,col=1)

    # Stage breakdown
    if cm and gm:
        fig.add_trace(go.Bar(name="pre",   x=["CPU","GPU"], y=[cm[0], gm[0]]), row=1,col=2)
        fig.add_trace(go.Bar(name="infer", x=["CPU","GPU"], y=[cm[1], gm[1]]), row=1,col=2)
        fig.add_trace(go.Bar(name="post",  x=["CPU","GPU"], y=[cm[2], gm[2]]), row=1,col=2)

    # ECDF
    xc,yc = ecdf_xy(cpu["e2e_ms"]); xg,yg = ecdf_xy(gpu["e2e_ms"])
    if len(xc): fig.add_trace(go.Scatter(x=xc, y=yc, name="CPU", mode="lines"), row=2,col=1)
    if len(xg): fig.add_trace(go.Scatter(x=xg, y=yg, name="GPU", mode="lines"), row=2,col=1)

    # Per-frame latency
    fig.add_trace(go.Scatter(x=list(cpu.index), y=list(cpu["e2e_ms"]), name="CPU", mode="lines"), row=2,col=2)
    fig.add_trace(go.Scatter(x=list(gpu.index), y=list(gpu["e2e_ms"]), name="GPU", mode="lines"), row=2,col=2)

    # Rolling FPS
    fig.add_trace(go.Scatter(x=list(cpu.index), y=list(cpu_fps.rolling(30, min_periods=1).mean()),
                             name="CPU roll FPS", mode="lines"), row=3,col=1)
    fig.add_trace(go.Scatter(x=list(gpu.index), y=list(gpu_fps.rolling(30, min_periods=1).mean()),
                             name="GPU roll FPS", mode="lines"), row=3,col=1)

    # Power/Energy
    p_cpu = js_cpu["system_means"].get("power_mean_w") if js_cpu else None
    p_gpu = js_gpu["system_means"].get("power_mean_w") if js_gpu else None
    e_cpu = js_cpu.get("energy_per_frame_j") if js_cpu else None
    e_gpu = js_gpu.get("energy_per_frame_j") if js_gpu else None
    if p_cpu is not None or p_gpu is not None:
        fig.add_trace(go.Bar(x=["CPU","GPU"], y=[p_cpu or 0, p_gpu or 0], name="Power (W)"), row=3,col=2)
    if e_cpu is not None or e_gpu is not None:
        fig.add_trace(go.Bar(x=["CPU","GPU"], y=[e_cpu or 0, e_gpu or 0], name="Energy (J/frame)"), row=3,col=2)

    fig.update_layout(barmode="stack", height=1000, title="CPU vs GPU – YOLOv8 Benchmark Dashboard")
    ensure_dir(out)
    pio.write_html(fig, file=f"{out}/dashboard.html", auto_open=False, include_plotlyjs="cdn")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_csv", required=True)
    ap.add_argument("--gpu_csv", required=True)
    ap.add_argument("--cpu_json", default=None)
    ap.add_argument("--gpu_json", default=None)
    ap.add_argument("--out", default="viz_out")
    ap.add_argument("--lib", choices=["mpl","seaborn","plotly","all"], default="mpl",
                    help="Which library to use")
    args = ap.parse_args()

    ensure_dir(args.out)
    cpu, gpu, js_cpu, js_gpu = read_runs(args.cpu_csv, args.gpu_csv, args.cpu_json, args.gpu_json)

    if args.lib in ("mpl","all"):
        mpl_plots(cpu, gpu, js_cpu, js_gpu, args.out)
    if args.lib in ("seaborn","all"):
        seaborn_plots(cpu, gpu, js_cpu, js_gpu, args.out)
    if args.lib in ("plotly","all"):
        plotly_dashboard(cpu, gpu, js_cpu, js_gpu, args.out)

    print(f"Saved visualizations to: {args.out}")

if __name__ == "__main__":
    main()
