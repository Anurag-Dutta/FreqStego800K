import ast
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


BASE = Path(
    r"..."
)

OUT_DIR = BASE / "ablation_spectral_components"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "AI_ArtBench": "AI-ArtBench",
    "BOSSbase_1.01": "BOSSbase1.01",
    "BOWS2": "BOWS2",
    "animal_faces_hq_afhq": "AFHQ",
    "emonet_faces": "EmoNet-Faces",
    "happywhales": "HappyWhales",
    "landcover": "LandCover",
    "wikiart": "WikiArt",
}

LEVELS = {
    "low": 0.01,
    "modest": 0.05,
    "high": 0.10,
}

MAX_IMAGES_PER_DATASET = 3

EVALUATION_MANIFESTS = {}

REPRESENTATIVES = {
    "Fourier": "DFT_LSB",
    "Cosine": "DCT_LSB",
    "Wavelet": "DWT_LSB",
}

FAMILIES = list(REPRESENTATIVES)

BAND_SETTINGS = ["None", "Top-1", "Top-2"]

METRICS = ["psnr", "ssim", "ber"]

DIGITS = {
    "psnr": 2,
    "ssim": 4,
    "ber": 6,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONDITION_KEYS = [
    (band_setting, payload_encoding, bitplane_embedding)
    for band_setting in BAND_SETTINGS
    for payload_encoding, bitplane_embedding in [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
]


def load_spectral_namespace(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    namespace = {
        "__name__": "spectral_ablation_import",
        "__file__": str(notebook_path),
    }

    allowed = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    finished = False

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )

        if not source.strip():
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError as exception:
            raise RuntimeError(
                f"Cannot parse cell {cell_index} of "
                f"{notebook_path.name}: {exception}"
            ) from exception

        for node in tree.body:
            if not isinstance(node, allowed):
                continue

            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)

            exec(
                compile(module, str(notebook_path), "exec"),
                namespace,
            )

            if isinstance(node, ast.FunctionDef) and node.name == "main":
                finished = True
                break

        if finished:
            break

    required = [
        "SABSNet",
        "ALGO_SPEC",
        "ALGO_INDEX",
        "IMAGE_SIZE",
        "collect_cover_paths",
        "image_id_from_path",
        "load_cover",
        "rgb_to_luma",
        "build_catalogues",
        "random_payload_for_instance",
        "build_band_inputs",
        "embed_image",
        "decode_image",
        "embed_vector",
        "decode_vector",
        "compute_psnr",
        "compute_ssim",
        "compute_ber",
    ]

    missing = [name for name in required if name not in namespace]

    if missing:
        raise RuntimeError(
            f"{notebook_path.name}: missing definitions {missing}"
        )

    for algorithm in REPRESENTATIVES.values():
        if namespace["ALGO_SPEC"][algorithm]["scheme"] != "lsb":
            raise RuntimeError(
                f"{algorithm} is not an LSB scheme in this notebook."
            )

    namespace["DEVICE"] = DEVICE
    return namespace


def load_checkpoint(namespace, path):
    model = namespace["SABSNet"]().to(DEVICE)

    state = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def find_evaluation_covers(namespace, dataset_key, level_root):
    cover_directory = BASE / dataset_key

    all_paths = namespace["collect_cover_paths"](
        cover_directory,
        None,
    )

    path_by_id = {}

    for path in all_paths:
        image_id = namespace["image_id_from_path"](
            path,
            cover_directory,
        )

        if image_id in path_by_id:
            raise RuntimeError(
                f"Duplicate image identifier: {dataset_key}/{image_id}"
            )

        path_by_id[image_id] = path

    manifest = EVALUATION_MANIFESTS.get(dataset_key)

    if manifest is not None:
        frame = pd.read_csv(
            manifest,
            dtype={"image_id": str},
        )

        if "image_id" not in frame:
            raise ValueError(
                f"Missing image_id column: {manifest}"
            )

        requested_ids = set(frame["image_id"].dropna())

    else:
        files = sorted(level_root.glob("*/metadata.json"))

        if not files:
            raise FileNotFoundError(
                f"No generation metadata under {level_root}"
            )

        requested_ids = set()

        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                rows = json.load(handle)

            requested_ids.update(
                str(row["image_id"])
                for row in rows
                if row.get("image_id") is not None
            )

    missing = requested_ids - set(path_by_id)

    if missing:
        raise FileNotFoundError(
            f"{dataset_key}: unresolved cover IDs: "
            f"{sorted(missing)[:5]}"
        )

    selected = [
        (
            namespace["image_id_from_path"](
                path,
                cover_directory,
            ),
            path,
        )
        for path in all_paths
        if namespace["image_id_from_path"](
            path,
            cover_directory,
        ) in requested_ids
    ]

    if MAX_IMAGES_PER_DATASET is not None:
        selected = selected[:MAX_IMAGES_PER_DATASET]

    return selected


def scrambling_permutation(dataset_key, image_id, algorithm, n_bits):
    token = (
        f"42|scrambling_control|"
        f"{dataset_key}|{image_id}|{algorithm}"
    ).encode("utf-8")

    seed = int.from_bytes(
        hashlib.sha256(token).digest()[:8],
        byteorder="little",
        signed=False,
    )

    rng = np.random.default_rng(seed)

    return rng.permutation(n_bits), seed


def encode_payload(payload, permutation, enabled):
    if enabled:
        return payload[permutation].copy()

    return payload.copy()


def decode_payload(encoded_bits, permutation, enabled):
    if not enabled:
        return encoded_bits.copy()

    recovered = np.empty_like(encoded_bits)
    recovered[permutation] = encoded_bits

    return recovered


def nearest_parity_embed(values, bits, delta):
    magnitude = np.abs(values)
    normalized = magnitude / delta

    q = np.rint(normalized).astype(np.int64)
    mismatch = (q & 1) != bits.astype(np.int64)

    lower = np.maximum(q - 1, 0)
    upper = q + 1

    lower_distance = np.abs(normalized - lower)
    upper_distance = np.abs(normalized - upper)

    choose_lower = (
        (q > 0)
        & (lower_distance <= upper_distance)
    )

    adjusted = np.where(choose_lower, lower, upper)
    q = np.where(mismatch, adjusted, q)

    modified_magnitude = q * delta

    if np.iscomplexobj(values):
        phase_unit = np.ones_like(values, dtype=np.complex128)
        nonzero = magnitude > 0
        phase_unit[nonzero] = values[nonzero] / magnitude[nonzero]
        return modified_magnitude * phase_unit

    sign = np.where(values < 0, -1.0, 1.0)
    return modified_magnitude * sign


def nearest_parity_decode(values, delta):
    q = np.rint(
        np.abs(values) / delta
    ).astype(np.int64)

    return (q & 1).astype(np.uint8)


def learned_scores(namespace, model, descriptors, algorithm):
    inputs = torch.tensor(
        descriptors,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    algorithm_index = torch.tensor(
        [namespace["ALGO_INDEX"][algorithm]],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        scores, _, _, _ = model(inputs, algorithm_index)

    return scores[0].cpu().numpy()


def selected_bands(catalogue, scores, setting):
    if setting == "None":
        return list(range(len(catalogue.bands)))

    k = 1 if setting == "Top-1" else 2

    order = np.argsort(
        -np.asarray(scores),
        kind="stable",
    )

    return sorted(int(index) for index in order[:k])


def evaluate_variant(
    namespace,
    cover,
    original_payload,
    encoded_payload,
    permutation,
    payload_encoding,
    bitplane_embedding,
    algorithm,
    catalogue,
    band_ids,
    temporary_png,
):
    original_embed_vector = namespace["embed_vector"]
    original_decode_vector = namespace["decode_vector"]

    if not bitplane_embedding:
        def alternative_embed_vector(
            algorithm_name,
            values,
            partner_values,
            bits,
            delta,
        ):
            if algorithm_name != algorithm:
                return original_embed_vector(
                    algorithm_name,
                    values,
                    partner_values,
                    bits,
                    delta,
                )

            return (
                nearest_parity_embed(values, bits, delta),
                None,
                0,
            )

        def alternative_decode_vector(
            algorithm_name,
            values,
            partner_values,
            delta,
        ):
            if algorithm_name != algorithm:
                return original_decode_vector(
                    algorithm_name,
                    values,
                    partner_values,
                    delta,
                )

            return nearest_parity_decode(values, delta)

        namespace["embed_vector"] = alternative_embed_vector
        namespace["decode_vector"] = alternative_decode_vector

    try:
        result = namespace["embed_image"](
            algorithm,
            cover,
            encoded_payload,
            band_ids,
            catalogue,
        )

        Image.fromarray(result["stego"]).save(temporary_png)

        with Image.open(temporary_png) as image:
            saved_stego = np.array(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        encoded_decoded_channels = namespace["decode_image"](
            algorithm,
            saved_stego,
            len(encoded_payload),
            band_ids,
            catalogue,
            result["delta"],
            result["skip"],
        )

        decoded_channels = [
            decode_payload(
                bits,
                permutation,
                payload_encoding,
            )
            for bits in encoded_decoded_channels
        ]

        ber = max(
            namespace["compute_ber"](
                original_payload,
                bits,
            )
            for bits in decoded_channels
        )

        return {
            "psnr": float(
                namespace["compute_psnr"](cover, saved_stego)
            ),
            "ssim": float(
                namespace["compute_ssim"](cover, saved_stego)
            ),
            "ber": float(ber),
            "recovered_ok": bool(
                all(
                    np.array_equal(original_payload, bits)
                    for bits in decoded_channels
                )
            ),
            "delta": float(result["delta"]),
            "skip_count": len(result["skip"]),
            "n_boosted_pairs": int(result["n_boosted"]),
        }

    finally:
        namespace["embed_vector"] = original_embed_vector
        namespace["decode_vector"] = original_decode_vector


def run_dataset_level(namespace, dataset_key, pretty_name, level):
    level_root = (
        BASE
        / f"{dataset_key}_stego_spectral"
        / level
    )

    checkpoint_path = level_root / "sabsnet_weights.pt"
    model = load_checkpoint(namespace, checkpoint_path)

    covers = find_evaluation_covers(
        namespace,
        dataset_key,
        level_root,
    )

    width, height = namespace["IMAGE_SIZE"]
    bpp = LEVELS[level]
    n_bits = int(bpp * width * height)

    catalogues = namespace["build_catalogues"](
        height,
        width,
    )

    records = []

    with tempfile.TemporaryDirectory(
        prefix="component_ablation_"
    ) as directory:
        temporary_png = Path(directory) / "stego.png"

        for image_id, cover_path in tqdm(
            covers,
            desc=f"{pretty_name} / {level}",
        ):
            cover = namespace["load_cover"](cover_path)
            luminance = namespace["rgb_to_luma"](cover)
            transform_cache = {}

            for family, algorithm in REPRESENTATIVES.items():
                catalogue = catalogues[algorithm]

                payload, payload_seed = (
                    namespace["random_payload_for_instance"](
                        image_id,
                        algorithm,
                        n_bits,
                    )
                )

                permutation, scrambling_seed = scrambling_permutation(
                    dataset_key,
                    image_id,
                    algorithm,
                    n_bits,
                )

                descriptors = namespace["build_band_inputs"](
                    algorithm,
                    catalogue,
                    luminance,
                    payload,
                    transform_cache,
                )

                scores = learned_scores(
                    namespace,
                    model,
                    descriptors,
                    algorithm,
                )

                for setting, payload_encoding, bitplane_embedding in CONDITION_KEYS:
                    band_ids = selected_bands(
                        catalogue,
                        scores,
                        setting,
                    )

                    capacity = int(
                        catalogue.capacities[band_ids].sum()
                    )

                    record = {
                        "dataset": pretty_name,
                        "level": level,
                        "target_bpp": bpp,
                        "image_id": image_id,
                        "family": family,
                        "algorithm": algorithm,
                        "band_selection": setting,
                        "payload_encoding": payload_encoding,
                        "bitplane_embedding": bitplane_embedding,
                        "payload_length": n_bits,
                        "payload_seed": int(payload_seed),
                        "scrambling_seed": int(scrambling_seed),
                        "band_ids": json.dumps(band_ids),
                        "selected_capacity": capacity,
                        "status": "ok",
                        "error": "",
                    }

                    if capacity < n_bits:
                        record["status"] = "infeasible_capacity"
                        record["error"] = (
                            f"Capacity {capacity} < payload {n_bits}"
                        )
                        records.append(record)
                        continue

                    encoded_payload = encode_payload(
                        payload,
                        permutation,
                        payload_encoding,
                    )

                    try:
                        result = evaluate_variant(
                            namespace=namespace,
                            cover=cover,
                            original_payload=payload,
                            encoded_payload=encoded_payload,
                            permutation=permutation,
                            payload_encoding=payload_encoding,
                            bitplane_embedding=bitplane_embedding,
                            algorithm=algorithm,
                            catalogue=catalogue,
                            band_ids=band_ids,
                            temporary_png=temporary_png,
                        )

                        record.update(result)

                    except Exception as exception:
                        record["status"] = "error"
                        record["error"] = str(exception)

                    records.append(record)

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return records


GROUP_COLUMNS = [
    "level",
    "family",
    "band_selection",
    "payload_encoding",
    "bitplane_embedding",
]


def numeric_summary(frame, grouping):
    groups = frame.groupby(
        grouping,
        sort=False,
        dropna=False,
    )

    counts = groups["status"].agg(
        n_attempts="size",
        n_generated=lambda values: int(values.eq("ok").sum()),
        n_infeasible=lambda values: int(
            values.eq("infeasible_capacity").sum()
        ),
        n_errors=lambda values: int(values.eq("error").sum()),
    )

    valid = frame[frame["status"].eq("ok")].copy()

    if valid.empty:
        return counts.reset_index()

    g = valid.groupby(
        grouping,
        sort=False,
        dropna=False,
    )

    finite = valid.copy()

    for metric in METRICS:
        finite[metric] = finite[metric].where(
            np.isfinite(finite[metric]),
            np.nan,
        )

    fg = finite.groupby(
        grouping,
        sort=False,
        dropna=False,
    )

    means = fg[METRICS].mean().add_suffix("_mean")
    stds = fg[METRICS].std(ddof=1).add_suffix("_std")
    metric_counts = fg[METRICS].count().add_suffix("_n")

    recovery = (
        g["recovered_ok"].mean() * 100.0
    ).rename("exact_recovery_percent")

    infinity_counts = g["psnr"].apply(
        lambda values: int(np.isposinf(values).sum())
    ).rename("n_infinite_psnr")

    result = counts.join(
        [means, stds, metric_counts, recovery, infinity_counts]
    )

    result["mean_delta"] = g["delta"].mean()
    result["mean_skip_count"] = g["skip_count"].mean()

    return result.reset_index()


def format_mean_std(mean, std, metric, latex=False):
    if pd.isna(mean):
        return "--"

    def number(value):
        if metric == "ber" and value != 0 and abs(value) < 1e-6:
            if latex:
                exponent = int(np.floor(np.log10(abs(value))))
                coefficient = value / (10.0 ** exponent)
                return (
                    rf"${coefficient:.2f}\times10^{{{exponent}}}$"
                )
            return f"{value:.2e}"

        return f"{value:.{DIGITS[metric]}f}"

    mean_text = number(float(mean))

    if pd.isna(std):
        return mean_text

    std_text = number(float(std))

    if latex:
        return (
            mean_text
            + r" {\fontsize{6}{8}\selectfont $\pm$ "
            + std_text
            + "}"
        )

    return f"{mean_text} ± {std_text}"


def lookup(summary, level, family, key):
    setting, payload_encoding, bitplane_embedding = key

    rows = summary[
        summary["level"].eq(level)
        & summary["family"].eq(family)
        & summary["band_selection"].eq(setting)
        & summary["payload_encoding"].eq(payload_encoding)
        & summary["bitplane_embedding"].eq(bitplane_embedding)
    ]

    return None if rows.empty else rows.iloc[0]


def table_frame(summary, level, metric):
    records = []

    for key in CONDITION_KEYS:
        setting, payload_encoding, bitplane_embedding = key

        row = {
            "Band Selection": setting,
            "Payload Encoding": "✓" if payload_encoding else "×",
            "Bit-plane Embedding": "✓" if bitplane_embedding else "×",
        }

        for family in FAMILIES:
            values = lookup(summary, level, family, key)

            row[family] = (
                "--"
                if values is None
                else format_mean_std(
                    values.get(f"{metric}_mean", np.nan),
                    values.get(f"{metric}_std", np.nan),
                    metric,
                )
            )

        records.append(row)

    return pd.DataFrame(records)


def latex_table(summary, level, metric):
    heading = "PSNR (dB)" if metric == "psnr" else metric.upper()

    lines = [
        r"\begin{table}[!t]",
        (
            rf"\caption{{\label{{tab:component_{level}_{metric}}}"
            rf" Frozen-selector encoder-control ablation at "
            rf"{LEVELS[level]} bpp. Mean $\pm$ sample standard "
            rf"deviation of {heading}.}}"
        ),
        r"\centering",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{|c|c|c|c|c|c|}",
        r"\hline",
        (
            r"\multirow{2}{*}{\shortstack{\textbf{Band}\\"
            r"\textbf{Selection}}} & "
            r"\multirow{2}{*}{\shortstack{\textbf{Payload}\\"
            r"\textbf{Encoding}}} & "
            r"\multirow{2}{*}{\shortstack{\textbf{Bit-plane}\\"
            r"\textbf{Embedding}}} & "
            rf"\multicolumn{{3}}{{c|}}{{\textbf{{Embedding Family, {heading}}}}}"
            r" \\ \cline{4-6}"
        ),
        (
            r" & & & \textbf{Fourier} & \textbf{Cosine} & "
            r"\textbf{Wavelet} \\ \hline\hline"
        ),
    ]

    for setting in BAND_SETTINGS:
        if setting == "None":
            lines.append(
                r"\multicolumn{6}{|c|}{\cellcolor{gray!20}"
                r"\textit{Without SABSNet---No band selection}}"
                r" \\ \hline"
            )

        elif setting == "Top-1":
            lines.append(
                r"\multicolumn{6}{|c|}{\cellcolor{gray!20}"
                r"\textit{With SABSNet for band selection}}"
                r" \\ \hline"
            )

        settings = [
            key
            for key in CONDITION_KEYS
            if key[0] == setting
        ]

        for index, key in enumerate(settings):
            _, payload_encoding, bitplane_embedding = key

            first = (
                rf"\multirow{{4}}{{*}}{{{setting}}}"
                if index == 0
                else ""
            )

            cells = []

            for family in FAMILIES:
                values = lookup(
                    summary,
                    level,
                    family,
                    key,
                )

                cells.append(
                    "--"
                    if values is None
                    else format_mean_std(
                        values.get(f"{metric}_mean", np.nan),
                        values.get(f"{metric}_std", np.nan),
                        metric,
                        latex=True,
                    )
                )

            payload_mark = (
                r"$\checkmark$"
                if payload_encoding
                else r"$\times$"
            )

            bitplane_mark = (
                r"$\checkmark$"
                if bitplane_embedding
                else r"$\times$"
            )

            ending = (
                r" \\ \hline"
                if index == 3
                else r" \\ \cline{2-6}"
            )

            lines.append(
                f"{first} & {payload_mark} & {bitplane_mark} & "
                + " & ".join(cells)
                + ending
            )

    lines.extend([
        r"\end{tabular}%",
        "}",
        r"\par\smallskip",
        (
            r"\scriptsize "
            r"Fixed representatives: DFT-LSB, DCT-LSB and DWT-LSB. "
            r"Payload encoding denotes reversible seeded scrambling; "
            r"its absence uses the original bit order. "
            r"Without bit-plane replacement, nearest-parity magnitude "
            r"quantization is used instead. "
            r"The original encoder corresponds to payload encoding "
            r"$\times$ and bit-plane embedding $\checkmark$. "
            r"The selector is frozen and shared across encoder controls. "
            r"Infeasible/error cases are counted separately; finite "
            r"PSNR means exclude infinite values."
        ),
        r"\end{table}",
    ])

    return "\n".join(lines)


def export_results(records):
    frame = pd.DataFrame(records)

    numeric_columns = [
        "psnr",
        "ssim",
        "ber",
        "delta",
        "skip_count",
        "n_boosted_pairs",
    ]

    for column in numeric_columns:
        if column not in frame:
            frame[column] = np.nan

    if "recovered_ok" not in frame:
        frame["recovered_ok"] = np.nan

    frame.to_csv(
        OUT_DIR / "component_ablation_per_image.csv",
        index=False,
    )

    summary = numeric_summary(
        frame,
        GROUP_COLUMNS,
    )

    dataset_summary = numeric_summary(
        frame,
        ["dataset", "algorithm"] + GROUP_COLUMNS,
    )

    summary.to_csv(
        OUT_DIR / "component_ablation_family_summary.csv",
        index=False,
    )

    dataset_summary.to_csv(
        OUT_DIR / "component_ablation_dataset_summary.csv",
        index=False,
    )

    for level in LEVELS:
        if level not in set(frame["level"]):
            continue

        for metric in METRICS:
            table_frame(summary, level, metric).to_csv(
                OUT_DIR / f"component_{level}_{metric}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            (OUT_DIR / f"component_{level}_{metric}.tex").write_text(
                latex_table(summary, level, metric),
                encoding="utf-8",
            )

    return frame, summary


def main():
    all_records = []

    protocol = {
        "datasets": DATASETS,
        "levels": LEVELS,
        "max_images_per_dataset": MAX_IMAGES_PER_DATASET,
        "representatives": REPRESENTATIVES,
        "selector": "saved checkpoint, frozen",
        "selection_features": "original payload and original encoder",
        "payload_encoding_on": "reversible seeded permutation",
        "payload_encoding_off": "original bit order",
        "bitplane_on": "original magnitude bit-plane replacement",
        "bitplane_off": "nearest-parity magnitude lattice encoder",
        "original_condition": {
            "payload_encoding": False,
            "bitplane_embedding": True,
        },
    }

    with open(
        OUT_DIR / "ablation_protocol.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(protocol, handle, indent=2)

    for dataset_key, pretty_name in DATASETS.items():
        notebook_path = BASE / f"{dataset_key}.ipynb"

        if not notebook_path.exists():
            print(f"[missing notebook] {notebook_path}")
            continue

        namespace = load_spectral_namespace(
            notebook_path
        )

        for level in LEVELS:
            checkpoint_path = (
                BASE
                / f"{dataset_key}_stego_spectral"
                / level
                / "sabsnet_weights.pt"
            )

            if not checkpoint_path.exists():
                print(f"[missing checkpoint] {checkpoint_path}")
                continue

            records = run_dataset_level(
                namespace,
                dataset_key,
                pretty_name,
                level,
            )

            all_records.extend(records)
            export_results(all_records)

    if not all_records:
        raise RuntimeError(
            "No results generated. Check notebooks, "
            "checkpoints and generation metadata."
        )

    frame, summary = export_results(
        all_records
    )

    for level in LEVELS:
        if level not in set(frame["level"]):
            continue

        print("\n" + "=" * 100)
        print(f"COMPONENT CONTROLS: {level}, {LEVELS[level]} bpp")
        print("=" * 100)

        for metric in METRICS:
            print(f"\n{metric.upper()} — mean ± std")
            print(
                table_frame(
                    summary,
                    level,
                    metric,
                ).to_string(index=False)
            )

        counts = [
            "family",
            "band_selection",
            "payload_encoding",
            "bitplane_embedding",
            "n_attempts",
            "n_generated",
            "n_infeasible",
            "n_errors",
        ]

        for column in [
            "exact_recovery_percent",
            "n_infinite_psnr",
        ]:
            if column in summary:
                counts.append(column)

        print("\nCoverage and recovery:")
        print(
            summary[
                summary["level"].eq(level)
            ][counts].to_string(index=False)
        )

    print(f"\nSaved to: {OUT_DIR}")

    return frame, summary


if __name__ == "__main__":
    ablation_rows, ablation_summary = main()