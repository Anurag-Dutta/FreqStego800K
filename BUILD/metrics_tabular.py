import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(r"...")
OUT_DIR = BASE / "metric_summaries"


DATASETS = {
    "AI_ArtBench_stego_spectral":          "AI-ArtBench",
    "BOSSbase_1.01_stego_spectral":        "BOSSbase1.01",
    "BOWS2_stego_spectral":                "BOWS2",
    "animal_faces_hq_afhq_stego_spectral": "AFHQ",
    "emonet_faces_stego_spectral":         "EmoNet-Faces",
    "happywhales_stego_spectral":          "HappyWhales",
    "landcover_stego_spectral":            "LandCover",
    "wikiart_stego_spectral":              "WikiArt",
}
ALGOS = ["DCT_LSB", "DCT_PHASE", "DCT_ZIGZAG", "DFT_LSB", "DFT_PHASE",
         "DFT_RADIAL", "DST_LSB", "DWT_LSB", "DWT_BP4", "DWT_CROSS"]
LEVELS = ["low", "modest", "high"]
METRICS = ["psnr", "snr", "ssim", "mae", "mse", "ber"]
DIGITS = {"psnr": 2, "snr": 2, "ssim": 4, "mae": 4, "mse": 4, "ber": 6}
MAIN_METRICS = [("psnr", "PSNR"), ("ssim", "SSIM"), ("ber", "BER")]
CSV_NAME = "metrics_all_levels.csv"



def load_all():
    frames = []
    for folder, name in DATASETS.items():
        path = BASE / folder / CSV_NAME
        if not path.exists():
            print(f"[missing] {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "dataset", name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No metrics_all_levels.csv found.")
    df = pd.concat(frames, ignore_index=True)
    for c in METRICS + ["delta", "n_boosted_pairs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["recovered_ok"] = df["recovered_ok"].astype(str).str.lower().eq("true")
    df["failed"] = df["error"].notna() | (df["psnr"].isna() & df["ber"].isna())
    return df



def summarise(df):
    ok = df[~df["failed"]]
    g = ok.groupby(["dataset", "level", "algorithm"])
    out = g[METRICS].mean().add_suffix("_mean")
    out = out.join(g[METRICS].std().add_suffix("_std"))
    out["n_images"] = g.size()
    out["exact_recovery_%"] = g["recovered_ok"].mean() * 100
    out["mean_delta"] = g["delta"].mean()
    out["mean_boosted_pairs"] = g["n_boosted_pairs"].mean()
    out["n_failed_rows"] = df.groupby(["dataset", "level", "algorithm"])["failed"].sum()
    out = out.reset_index()
    out["dataset"] = pd.Categorical(out["dataset"], list(DATASETS.values()), ordered=True)
    out["algorithm"] = pd.Categorical(out["algorithm"], ALGOS, ordered=True)
    out["level"] = pd.Categorical(out["level"], LEVELS, ordered=True)
    return out.sort_values(["level", "dataset", "algorithm"]).reset_index(drop=True)



def pivot_mean_std(df, level, metric):
    ok = df[(~df["failed"]) & (df["level"] == level)]
    ds, al = list(DATASETS.values()), ALGOS
    g = ok.groupby(["dataset", "algorithm"])[metric]
    mean = g.mean().unstack().reindex(index=ds, columns=al)
    std = g.std().unstack().reindex(index=ds, columns=al)

    pa = ok.groupby("algorithm")[metric].agg(["mean", "std"]).reindex(al)
    pd_ = ok.groupby("dataset")[metric].agg(["mean", "std"]).reindex(ds)
    mean.loc["Average"], std.loc["Average"] = pa["mean"], pa["std"]
    mean["Row mean"] = list(pd_["mean"]) + [ok[metric].mean()]
    std["Row mean"] = list(pd_["std"]) + [ok[metric].std()]
    return mean, std



def fmt(m, s, digits, latex=False):
    if pd.isna(m):
        return "--"
    if m == 0 and (pd.isna(s) or s == 0):
        m_txt, s_txt = "0", "0"
    else:
        m_txt = f"{m:.{digits}f}"
        s_txt = f"{s:.{digits}f}" if pd.notna(s) else None
    if latex:
        return m_txt if s_txt is None else (
            m_txt + r" {\fontsize{6}{8}\selectfont $\pm$ " + s_txt + "}")
    return m_txt if s_txt is None else f"{m_txt} ± {s_txt}"



def as_text(mean, std, digits, latex=False):
    out = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            out.loc[r, c] = fmt(mean.loc[r, c], std.loc[r, c], digits, latex)
    return out



def combined_tables(df, level, latex):
    return {lab: as_text(*pivot_mean_std(df, level, m), DIGITS[m], latex=latex)
            for m, lab in MAIN_METRICS}



def combined_csv(df, level):
    parts = []
    for lab, tbl in combined_tables(df, level, latex=False).items():
        t = tbl.copy()
        t.insert(0, "metric", lab)
        parts.append(t.reset_index())
    out = pd.concat(parts)
    order = list(DATASETS.values()) + ["Average"]
    out["dataset"] = pd.Categorical(out["dataset"], order, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], [l for _, l in MAIN_METRICS], ordered=True)
    return out.sort_values(["dataset", "metric"]).reset_index(drop=True)



def combined_latex(df, level, caption, label):
    tabs = combined_tables(df, level, latex=True)
    cols = list(next(iter(tabs.values())).columns)
    names = list(DATASETS.values()) + ["Average"]
    lines = [r"\begin{table*}[!t]", rf"\caption{{\label{{{label}}} {caption}}}",
             r"\renewcommand{\arraystretch}{1.32}", r"\centering", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{|c|c|" + "c|" * len(cols) + "}", r"\hline",
             r"\textbf{Dataset} & \textbf{Metric} & " + " & ".join(
                 r"\textbf{\textsc{" + c.replace("_", "-") + "}}" for c in cols) + r" \\ \hline\hline"]
    for n, name in enumerate(names):
        for k, (_, lab) in enumerate(MAIN_METRICS):
            first = rf"\multirow{{{len(MAIN_METRICS)}}}{{*}}{{\texttt{{{name}}}}}" if k == 0 else ""
            end = (r" \\ \hline" + (r"\hline" if name == names[-2] else "")
                   if k == len(MAIN_METRICS) - 1 else r" \\ \cline{2-" + str(len(cols) + 2) + "}")
            lines.append(f"{first} & \\texttt{{{lab}}} & " + " & ".join(tabs[lab].loc[name].tolist()) + end)
    lines += [r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines)



def latex_single(tbl, caption, label):
    cols = list(tbl.columns)
    lines = [r"\begin{table*}[!t]", rf"\caption{{\label{{{label}}} {caption}}}",
             r"\renewcommand{\arraystretch}{1.32}", r"\centering", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{|l|" + "c|" * len(cols) + "}", r"\hline",
             r"\textbf{Dataset} & " + " & ".join(
                 r"\textbf{\textsc{" + c.replace("_", "-") + "}}" for c in cols) + r" \\ \hline\hline"]
    for idx in tbl.index:
        sep = r" \\ \hline\hline" if idx == tbl.index[-2] else r" \\ \hline"
        lines.append(f"{idx} & " + " & ".join(tbl.loc[idx].tolist()) + sep)
    lines += [r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines)



if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    summary = summarise(df)
    summary.to_csv(OUT_DIR / "summary_dataset_level_algorithm.csv", index=False)

    for level in LEVELS:
        if level not in set(df["level"].astype(str)):
            continue
        combined_csv(df, level).to_csv(
            OUT_DIR / f"combined_{level}_psnr_ssim_ber.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / f"combined_{level}_psnr_ssim_ber.tex").write_text(
            combined_latex(df, level,
                           f"Mean $\\pm$ std of PSNR (dB), SSIM and BER at the {level} payload level.",
                           f"tab:combined_{level}"), encoding="utf-8")
        for m in METRICS:
            mean, std = pivot_mean_std(df, level, m)
            mean.to_csv(OUT_DIR / f"pivot_{level}_{m}_mean.csv", float_format="%.6f")
            std.to_csv(OUT_DIR / f"pivot_{level}_{m}_std.csv", float_format="%.6f")
            as_text(mean, std, DIGITS[m]).to_csv(
                OUT_DIR / f"pivot_{level}_{m}_mean_pm_std.csv", encoding="utf-8-sig")
            (OUT_DIR / f"{m}_{level}.tex").write_text(
                latex_single(as_text(mean, std, DIGITS[m], latex=True),
                             f"Mean $\\pm$ std of {m.upper()} at the {level} payload level.",
                             f"tab:{m}_{level}"), encoding="utf-8")

        print(f"\n=== level = {level}: PSNR (dB) / SSIM / BER, mean ± std ===")
        for lab, tbl in combined_tables(df, level, latex=False).items():
            print(f"\n-- {lab} --")
            print(tbl.to_string())

    print(f"\nSaved to {OUT_DIR}")