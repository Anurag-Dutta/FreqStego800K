# %% [markdown]
# ## Spectral Stego Generation (Low / Modest / High BPP)

# %%
import os
import glob
import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.fft import dctn, idctn, dstn, idstn
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F


# %% Configuration
BASE = Path(r"...")
COVER_DIR = BASE / "..."
OUTPUT_BASE = BASE / "..."
METRICS_CSV = OUTPUT_BASE / "metrics_all_levels.csv"

IMAGE_SIZE = (512, 512)  # width, height
MAX_IMAGES = 10000

# Only these three settings differ between payload levels.
LEVELS = {
    "low":    {"target_bpp": 0.01, "min_psnr_gap": 0.01},
    "modest": {"target_bpp": 0.05, "min_psnr_gap": 0.05},
    "high":   {"target_bpp": 0.10, "min_psnr_gap": 0.10},
}

PAYLOAD_MASTER_SEED = 20261003
RANDOM_SEED = 42

N_RINGS = 8
K_FRAC = 0.40

BASE_DELTA = 4.0
DELTA_SCALES = (1.0, 1.5, 2.0, 3.0)
PHASE_STEP = math.pi / 2.0
CROSS_MARGIN = 0.5
MAX_ROUNDS = 15
CAL_BITS = 512

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 4
STAGE1_MAX_SAMPLES = 400
STAGE2_MAX_SAMPLES = 150
STAGE2_CANDIDATE_SUBSETS = 6

D_RAW = 6
PE_DIM = 8
D_IN = D_RAW + PE_DIM
H_DIM = 128
N_HEADS = 4
D_HEAD = 32

LAMBDA_SSL = 1.0
LAMBDA_RANK = 0.5
LAMBDA_DIV = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_KEYS = ("psnr", "snr", "ssim", "mae", "mse")

ALGO_SPEC = {
    "DFT_LSB":    {"transform": "dft", "family": "ring",    "scheme": "lsb",    "plane": 0,    "order": "perm"},
    "DWT_LSB":    {"transform": "dwt", "family": "subband", "scheme": "lsb",    "plane": 0,    "order": "perm"},
    "DCT_LSB":    {"transform": "dct", "family": "ring",    "scheme": "lsb",    "plane": 0,    "order": "perm"},
    "DST_LSB":    {"transform": "dst", "family": "ring",    "scheme": "lsb",    "plane": 0,    "order": "perm"},
    "DWT_BP4":    {"transform": "dwt", "family": "subband", "scheme": "lsb",    "plane": 4,    "order": "perm"},
    "DFT_PHASE":  {"transform": "dft", "family": "ring",    "scheme": "phase",  "plane": None, "order": "perm"},
    "DCT_PHASE":  {"transform": "dct", "family": "ring",    "scheme": "sign",   "plane": None, "order": "perm"},
    "DWT_CROSS":  {"transform": "dwt", "family": "pair",    "scheme": "cross",  "plane": None, "order": "perm"},
    "DFT_RADIAL": {"transform": "dft", "family": "ring",    "scheme": "lsb",    "plane": 1,    "order": "radial"},
    "DCT_ZIGZAG": {"transform": "dct", "family": "ring",    "scheme": "parity", "plane": None, "order": "zigzag"},
}

ALGO_LIST = list(ALGO_SPEC.keys())
ALGO_INDEX = {algorithm: index for index, algorithm in enumerate(ALGO_LIST)}


# %% General utilities
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_algorithm_dirs(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    directories = {}

    for algorithm in ALGO_LIST:
        directory = output_root / algorithm.lower()
        directory.mkdir(parents=True, exist_ok=True)
        directories[algorithm] = directory

    return directories


def collect_cover_paths(cover_directory, max_images):
    patterns = (
        "*.jpg", "*.jpeg", "*.png", "*.pgm", "*.bmp", "*.tif", "*.tiff",
        "*.JPG", "*.JPEG", "*.PNG", "*.PGM", "*.BMP", "*.TIF", "*.TIFF",
    )

    paths = []

    for pattern in patterns:
        paths.extend(
            glob.glob(
                os.path.join(str(cover_directory), "**", pattern),
                recursive=True,
            )
        )

    paths = sorted(set(paths))

    if max_images is not None:
        paths = paths[:max_images]

    return paths


def image_id_from_path(cover_path, cover_directory):
    relative_path = os.path.relpath(cover_path, str(cover_directory))

    return relative_path.replace(os.sep, "_").rsplit(".", 1)[0]


def load_cover(cover_path):
    image = Image.open(cover_path).convert("RGB")

    if image.size != IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE, Image.LANCZOS)

    return np.array(image, dtype=np.uint8)


def rgb_to_luma(rgb):
    return (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    )


def to_uint8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def compute_psnr(reference, candidate):
    mse = np.mean(
        (reference.astype(np.float64) - candidate.astype(np.float64)) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_snr(reference, candidate):
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)

    noise_power = np.sum((reference - candidate) ** 2)

    if noise_power == 0:
        return float("inf")

    return float(10.0 * np.log10(np.sum(reference ** 2) / noise_power))


def compute_mse(reference, candidate):
    return float(
        np.mean(
            (reference.astype(np.float64) - candidate.astype(np.float64)) ** 2
        )
    )


def compute_mae(reference, candidate):
    return float(
        np.mean(
            np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        )
    )


def compute_ssim(reference, candidate, data_range=255.0, sigma=1.5,
                 k1=0.01, k2=0.03):
    # Gaussian-window SSIM (Wang et al., 2004), averaged over RGB channels.
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)

    if reference.ndim == 2:
        reference = reference[..., None]
        candidate = candidate[..., None]

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    values = []

    for channel in range(reference.shape[-1]):
        x = reference[..., channel]
        y = candidate[..., channel]

        mu_x = gaussian_filter(x, sigma, truncate=3.5)
        mu_y = gaussian_filter(y, sigma, truncate=3.5)

        var_x = gaussian_filter(x * x, sigma, truncate=3.5) - mu_x ** 2
        var_y = gaussian_filter(y * y, sigma, truncate=3.5) - mu_y ** 2
        cov_xy = gaussian_filter(x * y, sigma, truncate=3.5) - mu_x * mu_y

        ssim_map = (
            (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        ) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        )

        values.append(float(np.mean(ssim_map)))

    return float(np.mean(values))


def compute_image_metrics(reference, candidate):
    return {
        "psnr": compute_psnr(reference, candidate),
        "snr": compute_snr(reference, candidate),
        "ssim": compute_ssim(reference, candidate),
        "mae": compute_mae(reference, candidate),
        "mse": compute_mse(reference, candidate),
    }


def round_or_none(value, digits):
    return round(float(value), digits) if np.isfinite(value) else None


def compute_ber(reference_bits, decoded_bits):
    reference_bits = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
    decoded_bits = np.asarray(decoded_bits, dtype=np.uint8).reshape(-1)

    if len(reference_bits) != len(decoded_bits):
        raise ValueError(
            "Reference and decoded bit vectors must have equal length."
        )

    return float(np.mean(reference_bits != decoded_bits))


# %% Payload utilities
def stable_payload_seed(image_id, algorithm, master_seed=PAYLOAD_MASTER_SEED):
    token = (f"{master_seed}|{image_id}|{algorithm}").encode("utf-8")
    digest = hashlib.sha256(token).digest()

    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def random_payload_for_instance(image_id, algorithm, n_bits):
    seed = stable_payload_seed(image_id=image_id, algorithm=algorithm)
    rng = np.random.default_rng(seed)

    payload = rng.integers(low=0, high=2, size=n_bits, dtype=np.uint8)

    return payload, seed


def payload_sha256(payload):
    packed = np.packbits(np.asarray(payload, dtype=np.uint8)).tobytes()

    return hashlib.sha256(packed).hexdigest()


# %% Two-level Haar DWT
SQRT2 = math.sqrt(2.0)


def haar_1d_forward(array, axis):
    even = np.take(array, np.arange(0, array.shape[axis], 2), axis=axis)
    odd = np.take(array, np.arange(1, array.shape[axis], 2), axis=axis)

    return (even + odd) / SQRT2, (even - odd) / SQRT2


def haar_1d_inverse(low, high, axis):
    shape = list(low.shape)
    shape[axis] *= 2

    output = np.empty(shape, dtype=low.dtype)

    even_slice = [slice(None)] * low.ndim
    odd_slice = [slice(None)] * low.ndim

    even_slice[axis] = slice(0, None, 2)
    odd_slice[axis] = slice(1, None, 2)

    output[tuple(even_slice)] = (low + high) / SQRT2
    output[tuple(odd_slice)] = (low - high) / SQRT2

    return output


def haar2_level_forward(array):
    low_rows, high_rows = haar_1d_forward(array, axis=0)
    ll, lh = haar_1d_forward(low_rows, axis=1)
    hl, hh = haar_1d_forward(high_rows, axis=1)

    return ll, lh, hl, hh


def haar2_level_inverse(ll, lh, hl, hh):
    low_rows = haar_1d_inverse(ll, lh, axis=1)
    high_rows = haar_1d_inverse(hl, hh, axis=1)

    return haar_1d_inverse(low_rows, high_rows, axis=0)


def dwt2_forward(array):
    height, width = array.shape

    if height % 4 != 0 or width % 4 != 0:
        raise ValueError("DWT input dimensions must be divisible by 4.")

    ll1, lh1, hl1, hh1 = haar2_level_forward(array)
    ll2, lh2, hl2, hh2 = haar2_level_forward(ll1)

    output = np.zeros_like(array, dtype=np.float64)

    half_h = height // 2
    half_w = width // 2
    quarter_h = height // 4
    quarter_w = width // 4

    output[:quarter_h, :quarter_w] = ll2
    output[:quarter_h, quarter_w:2 * quarter_w] = lh2
    output[quarter_h:2 * quarter_h, :quarter_w] = hl2
    output[quarter_h:2 * quarter_h, quarter_w:2 * quarter_w] = hh2

    output[:half_h, half_w:] = lh1
    output[half_h:, :half_w] = hl1
    output[half_h:, half_w:] = hh1

    return output


def dwt2_inverse(coefficients):
    height, width = coefficients.shape

    half_h = height // 2
    half_w = width // 2
    quarter_h = height // 4
    quarter_w = width // 4

    ll2 = coefficients[:quarter_h, :quarter_w]
    lh2 = coefficients[:quarter_h, quarter_w:2 * quarter_w]
    hl2 = coefficients[quarter_h:2 * quarter_h, :quarter_w]
    hh2 = coefficients[quarter_h:2 * quarter_h, quarter_w:2 * quarter_w]

    lh1 = coefficients[:half_h, half_w:]
    hl1 = coefficients[half_h:, :half_w]
    hh1 = coefficients[half_h:, half_w:]

    ll1 = haar2_level_inverse(ll2, lh2, hl2, hh2)

    return haar2_level_inverse(ll1, lh1, hl1, hh1)


# %% Transform utilities
def transform_forward(transform_name, channel):
    if transform_name == "dft":
        return np.fft.fft2(channel, norm="ortho")

    if transform_name == "dct":
        return dctn(channel, type=2, norm="ortho")

    if transform_name == "dst":
        return dstn(channel, type=2, norm="ortho")

    if transform_name == "dwt":
        return dwt2_forward(channel)

    raise ValueError(f"Unknown transform: {transform_name}")


def transform_inverse(transform_name, coefficients):
    if transform_name == "dft":
        return np.fft.ifft2(coefficients, norm="ortho").real

    if transform_name == "dct":
        return idctn(coefficients, type=2, norm="ortho")

    if transform_name == "dst":
        return idstn(coefficients, type=2, norm="ortho")

    if transform_name == "dwt":
        return dwt2_inverse(coefficients)

    raise ValueError(f"Unknown transform: {transform_name}")


# %% Band catalogue
def rectangle_flat_indices(row_range, col_range, width):
    rows = np.arange(row_range[0], row_range[1])[:, None]
    cols = np.arange(col_range[0], col_range[1])[None, :]

    return (rows * width + cols).reshape(-1).astype(np.int64)


class BandCatalogue:
    def __init__(self, algorithm, height, width, n_rings=N_RINGS):
        specification = ALGO_SPEC[algorithm]

        self.algorithm = algorithm
        self.transform = specification["transform"]
        self.family = specification["family"]
        self.height = height
        self.width = width
        self.bands = []

        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )

        flat = (rows * width + cols).astype(np.int64)

        if self.family == "ring":
            edges = np.linspace(0.0, 1.0, n_rings + 1)

            if self.transform == "dft":
                ky = np.where(rows < height // 2, rows, rows - height).astype(np.float64)
                kx = np.where(cols < width // 2, cols, cols - width).astype(np.float64)

                radius = np.hypot(
                    ky / (height / 2.0), kx / (width / 2.0)
                ) / math.sqrt(2.0)

                mate_rows = (-rows) % height
                mate_cols = (-cols) % width

                mate_grid = (mate_rows * width + mate_cols).astype(np.int64)

                self.mate = mate_grid.reshape(-1)

                # One coefficient from each Hermitian pair only.
                eligible = flat < mate_grid

            else:
                offset = 0 if self.transform == "dct" else 1

                radius = np.hypot(
                    (rows + offset) / (height - 1 + offset),
                    (cols + offset) / (width - 1 + offset),
                ) / math.sqrt(2.0)

                eligible = np.ones((height, width), dtype=bool)

                if self.transform == "dct":
                    eligible[0, 0] = False

            for band_id in range(n_rings):
                low = edges[band_id]
                high = edges[band_id + 1]

                if band_id < n_rings - 1:
                    mask = (radius >= low) & (radius < high)
                else:
                    mask = (radius >= low) & (radius <= high + 1e-12)

                self.bands.append(
                    {
                        "id": band_id,
                        "kind": "ring",
                        "rho": float((low + high) / 2.0),
                        "elig": flat[mask & eligible].astype(np.int64),
                        "partner": None,
                    }
                )

        elif self.family == "subband":
            half_h = height // 2
            half_w = width // 2
            quarter_h = height // 4
            quarter_w = width // 4

            regions = [
                (2, "LL", (0, quarter_h), (0, quarter_w)),
                (2, "LH", (0, quarter_h), (quarter_w, 2 * quarter_w)),
                (2, "HL", (quarter_h, 2 * quarter_h), (0, quarter_w)),
                (2, "HH", (quarter_h, 2 * quarter_h), (quarter_w, 2 * quarter_w)),
                (1, "LH", (0, half_h), (half_w, width)),
                (1, "HL", (half_h, height), (0, half_w)),
                (1, "HH", (half_h, height), (half_w, width)),
            ]

            for band_id, (level, band_type, row_range, col_range) in enumerate(regions):
                self.bands.append(
                    {
                        "id": band_id,
                        "kind": "subband",
                        "level": level,
                        "btype": band_type,
                        "elig": rectangle_flat_indices(row_range, col_range, width),
                        "partner": None,
                    }
                )

        elif self.family == "pair":
            half_h = height // 2
            half_w = width // 2
            quarter_h = height // 4
            quarter_w = width // 4

            pairs = [
                (2, (0, quarter_h), (quarter_w, 2 * quarter_w),
                    (quarter_h, 2 * quarter_h), (0, quarter_w)),
                (1, (0, half_h), (half_w, width),
                    (half_h, height), (0, half_w)),
            ]

            for band_id, (level, rows_a, cols_a, rows_b, cols_b) in enumerate(pairs):
                self.bands.append(
                    {
                        "id": band_id,
                        "kind": "subband",
                        "level": level,
                        "btype": "LH",
                        "elig": rectangle_flat_indices(rows_a, cols_a, width),
                        "partner": rectangle_flat_indices(rows_b, cols_b, width),
                    }
                )

        else:
            raise ValueError(f"Unknown family: {self.family}")

        self.capacities = np.asarray(
            [len(band["elig"]) for band in self.bands], dtype=np.int64
        )

    def get_slots(self, band_ids):
        band_ids = sorted(int(band_id) for band_id in band_ids)

        eligible = np.concatenate([self.bands[b]["elig"] for b in band_ids])

        partner = None

        if self.family == "pair":
            partner = np.concatenate([self.bands[b]["partner"] for b in band_ids])

        return eligible, partner


def build_catalogues(height, width):
    return {
        algorithm: BandCatalogue(algorithm, height, width)
        for algorithm in ALGO_LIST
    }


# %% Embedding and decoding primitives
def sign_or_positive(values):
    return np.where(values < 0.0, -1.0, 1.0)


def embed_lsb_lattice(values, bits, bit_plane, delta):
    magnitude = np.abs(values)

    indices = np.rint(magnitude / delta).astype(np.int64)

    indices = (indices & ~(1 << bit_plane)) | (bits.astype(np.int64) << bit_plane)

    modified_magnitude = indices * delta

    if np.iscomplexobj(values):
        unit = np.where(
            magnitude > 0.0,
            values / np.maximum(magnitude, 1e-300),
            1.0 + 0.0j,
        )

        return modified_magnitude * unit

    return modified_magnitude * sign_or_positive(values)


def decode_lsb_lattice(values, bit_plane, delta):
    indices = np.rint(np.abs(values) / delta).astype(np.int64)

    return ((indices >> bit_plane) & 1).astype(np.uint8)


def embed_parity_lattice(values, bits, delta):
    normalized = values / delta
    indices = np.rint(normalized).astype(np.int64)

    mismatch = (indices & 1) != bits.astype(np.int64)

    direction = np.where(normalized >= indices, 1, -1)

    return (indices + mismatch * direction) * delta


def decode_parity_lattice(values, delta):
    indices = np.rint(values / delta).astype(np.int64)

    return (indices & 1).astype(np.uint8)


def embed_sign(values, bits, delta):
    magnitude = np.maximum(np.abs(values), delta)

    return np.where(bits == 1, -magnitude, magnitude)


def decode_sign(values):
    return (values < 0.0).astype(np.uint8)


def embed_phase(values, bits, delta):
    magnitude = np.maximum(np.abs(values), delta)

    phase_values = np.angle(values) / PHASE_STEP
    phase_indices = np.rint(phase_values).astype(np.int64)

    mismatch = (phase_indices & 1) != bits.astype(np.int64)

    direction = np.where(phase_values >= phase_indices, 1, -1)

    phase_indices = phase_indices + mismatch * direction

    return magnitude * np.exp(1j * phase_indices * PHASE_STEP)


def decode_phase(values):
    phase_indices = np.rint(np.angle(values) / PHASE_STEP).astype(np.int64)

    return (phase_indices & 1).astype(np.uint8)


def embed_cross_subband(values_a, values_b, bits, margin):
    abs_a = np.abs(values_a)
    abs_b = np.abs(values_b)

    signs = np.where(bits == 1, 1.0, -1.0)

    valid = (signs * (abs_a - abs_b)) >= margin

    radius = np.hypot(values_a, values_b)

    repaired_radius = np.maximum(radius, margin)

    total_magnitude = np.sqrt(
        np.maximum(2.0 * repaired_radius ** 2 - margin ** 2, 0.0)
    )

    new_abs_a = (total_magnitude + signs * margin) / 2.0
    new_abs_b = (total_magnitude - signs * margin) / 2.0

    output_a = np.where(valid, values_a, sign_or_positive(values_a) * new_abs_a)
    output_b = np.where(valid, values_b, sign_or_positive(values_b) * new_abs_b)

    boosted = int(np.sum((~valid) & (radius < margin)))

    return output_a, output_b, boosted


def decode_cross_subband(values_a, values_b):
    return (np.abs(values_a) > np.abs(values_b)).astype(np.uint8)


def embed_vector(algorithm, values, partner_values, bits, delta):
    scheme = ALGO_SPEC[algorithm]["scheme"]

    if scheme == "lsb":
        return (
            embed_lsb_lattice(values, bits, ALGO_SPEC[algorithm]["plane"], delta),
            None,
            0,
        )

    if scheme == "parity":
        return embed_parity_lattice(values, bits, delta), None, 0

    if scheme == "sign":
        return embed_sign(values, bits, delta), None, 0

    if scheme == "phase":
        return embed_phase(values, bits, delta), None, 0

    if scheme == "cross":
        new_a, new_b, boosted = embed_cross_subband(
            values, partner_values, bits, CROSS_MARGIN * delta
        )

        return new_a, new_b, boosted

    raise ValueError(f"Unknown scheme: {scheme}")


def decode_vector(algorithm, values, partner_values, delta):
    scheme = ALGO_SPEC[algorithm]["scheme"]

    if scheme == "lsb":
        return decode_lsb_lattice(values, ALGO_SPEC[algorithm]["plane"], delta)

    if scheme == "parity":
        return decode_parity_lattice(values, delta)

    if scheme == "sign":
        return decode_sign(values)

    if scheme == "phase":
        return decode_phase(values)

    if scheme == "cross":
        return decode_cross_subband(values, partner_values)

    raise ValueError(f"Unknown scheme: {scheme}")


# %% Coefficient traversal
def visit_order(algorithm, catalogue, eligible_indices):
    order_type = ALGO_SPEC[algorithm]["order"]

    if order_type == "perm":
        rng = np.random.default_rng([RANDOM_SEED, ALGO_INDEX[algorithm]])

        return rng.permutation(len(eligible_indices))

    rows = eligible_indices // catalogue.width
    cols = eligible_indices % catalogue.width

    if order_type == "zigzag":
        diagonals = rows + cols

        secondary = np.where(diagonals % 2 == 1, rows, -rows)

        return np.lexsort((secondary, diagonals))

    if order_type == "radial":
        ky = np.where(
            rows < catalogue.height // 2, rows, rows - catalogue.height
        ).astype(np.float64)

        kx = np.where(
            cols < catalogue.width // 2, cols, cols - catalogue.width
        ).astype(np.float64)

        return np.lexsort((np.arctan2(ky, kx), np.hypot(ky, kx)))

    raise ValueError(f"Unknown order: {order_type}")


def slot_positions(algorithm, catalogue, eligible_indices, n_bits, skip_positions=None):
    order = visit_order(algorithm, catalogue, eligible_indices)

    if len(order) < n_bits:
        raise ValueError(
            f"{algorithm}: capacity {len(order)} is below payload size {n_bits}."
        )

    positions = order[:n_bits].copy()

    if skip_positions is None or len(skip_positions) == 0:
        return positions

    skip_positions = np.asarray(skip_positions, dtype=np.int64)

    invalid = np.isin(positions, skip_positions)

    n_invalid = int(invalid.sum())

    if n_invalid == 0:
        return positions

    spare_positions = order[n_bits:]
    spare_positions = spare_positions[~np.isin(spare_positions, skip_positions)]

    if len(spare_positions) < n_invalid:
        raise ValueError(f"{algorithm}: insufficient spare slots.")

    positions[invalid] = spare_positions[:n_invalid]

    return positions


# %% Image embedding and decoding
def embed_channel(algorithm, channel, payload_bits, band_ids, catalogue, delta,
                  skip_positions=None):
    coefficients = transform_forward(catalogue.transform, channel)

    flat = coefficients.reshape(-1).copy()

    eligible, partners = catalogue.get_slots(band_ids)

    positions = slot_positions(
        algorithm, catalogue, eligible, len(payload_bits), skip_positions
    )

    selected = eligible[positions]

    selected_partners = partners[positions] if partners is not None else None

    modified, modified_partners, boosted = embed_vector(
        algorithm,
        flat[selected],
        flat[selected_partners] if selected_partners is not None else None,
        payload_bits,
        delta,
    )

    flat[selected] = modified

    if selected_partners is not None:
        flat[selected_partners] = modified_partners

    if catalogue.transform == "dft":
        flat[catalogue.mate[selected]] = np.conj(modified)

    reconstructed = transform_inverse(
        catalogue.transform, flat.reshape(coefficients.shape)
    )

    return reconstructed, boosted


def decode_channel(algorithm, stego_channel, n_bits, band_ids, catalogue, delta,
                   skip_positions=None):
    coefficients = transform_forward(
        catalogue.transform, stego_channel.astype(np.float64)
    )

    flat = coefficients.reshape(-1)

    eligible, partners = catalogue.get_slots(band_ids)

    positions = slot_positions(
        algorithm, catalogue, eligible, n_bits, skip_positions
    )

    selected = eligible[positions]

    selected_partners = partners[positions] if partners is not None else None

    return decode_vector(
        algorithm,
        flat[selected],
        flat[selected_partners] if selected_partners is not None else None,
        delta,
    )


def decode_image(algorithm, stego_rgb, n_bits, band_ids, catalogue, delta,
                 skip_positions=None):
    return [
        decode_channel(
            algorithm, stego_rgb[..., channel], n_bits, band_ids,
            catalogue, delta, skip_positions,
        )
        for channel in range(3)
    ]


def embed_image(algorithm, cover_rgb, payload_bits, band_ids, catalogue):
    eligible, _ = catalogue.get_slots(band_ids)

    best_result = None

    for scale in DELTA_SCALES:
        delta = BASE_DELTA * scale

        skip_positions = np.zeros(0, dtype=np.int64)

        for _ in range(MAX_ROUNDS):
            try:
                positions = slot_positions(
                    algorithm, catalogue, eligible, len(payload_bits), skip_positions
                )
            except ValueError:
                break

            channels = []
            boosted_total = 0

            for channel_index in range(3):
                reconstructed, boosted = embed_channel(
                    algorithm,
                    cover_rgb[..., channel_index].astype(np.float64),
                    payload_bits,
                    band_ids,
                    catalogue,
                    delta,
                    skip_positions,
                )

                channels.append(to_uint8(reconstructed))

                boosted_total += boosted

            stego_rgb = np.stack(channels, axis=-1)

            decoded_channels = decode_image(
                algorithm, stego_rgb, len(payload_bits), band_ids,
                catalogue, delta, skip_positions,
            )

            worst_ber = max(
                compute_ber(payload_bits, decoded) for decoded in decoded_channels
            )

            result = {
                "stego": stego_rgb,
                "ber": worst_ber,
                "delta": float(delta),
                "skip": [int(position) for position in skip_positions],
                "n_boosted": int(boosted_total),
            }

            if best_result is None or result["ber"] < best_result["ber"]:
                best_result = result

            if worst_ber == 0.0:
                return result

            failed = np.zeros(len(payload_bits), dtype=bool)

            for decoded in decoded_channels:
                failed |= decoded != payload_bits

            skip_positions = np.union1d(skip_positions, positions[failed])

    if best_result is None:
        raise RuntimeError(f"{algorithm}: embedding failed.")

    return best_result


# %% Band descriptors
def positional_encoding_ring(rho, dimension=PE_DIM):
    values = []

    for frequency in range(1, dimension // 2 + 1):
        values.append(math.sin(2.0 * math.pi * frequency * rho))
        values.append(math.cos(2.0 * math.pi * frequency * rho))

    return np.asarray(values[:dimension], dtype=np.float32)


def positional_encoding_subband(level, band_type, dimension=PE_DIM):
    values = [
        float(level == 1),
        float(level == 2),
        float(band_type == "LL"),
        float(band_type == "LH"),
        float(band_type == "HL"),
        float(band_type == "HH"),
    ]

    values += [0.0] * (dimension - len(values))

    return np.asarray(values[:dimension], dtype=np.float32)


def compute_band_descriptors(algorithm, catalogue, flat_coefficients, calibration_bits):
    features = []
    positions = []

    for band in catalogue.bands:
        eligible = band["elig"]
        partners = band["partner"]

        if partners is None:
            values = np.abs(flat_coefficients[eligible])
        else:
            values = np.concatenate(
                [
                    np.abs(flat_coefficients[eligible]),
                    np.abs(flat_coefficients[partners]),
                ]
            )

        energy = math.log1p(float(np.sum(values ** 2)))

        probabilities = values / (values.sum() + 1e-12)
        probabilities = probabilities[probabilities > 0.0]

        entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
        ) / math.log2(len(values) + 1.0)

        group_size = 16
        usable = (len(values) // group_size) * group_size

        if usable:
            local_variance = float(
                np.mean(np.var(values[:usable].reshape(-1, group_size), axis=1))
            )
        else:
            local_variance = float(np.var(values))

        standardized = (values - values.mean()) / (values.std() + 1e-8)

        skewness = float(np.mean(standardized ** 3))
        kurtosis = float(np.mean(standardized ** 4))

        n_calibration = min(CAL_BITS, len(eligible))

        local_positions = slot_positions(algorithm, catalogue, eligible, n_calibration)

        selected = eligible[local_positions]

        selected_partners = partners[local_positions] if partners is not None else None

        original = flat_coefficients[selected]

        original_partners = (
            flat_coefficients[selected_partners]
            if selected_partners is not None
            else None
        )

        modified, modified_partners, _ = embed_vector(
            algorithm,
            original,
            original_partners,
            calibration_bits[:n_calibration],
            BASE_DELTA,
        )

        error = float(np.sum(np.abs(modified - original) ** 2))

        if modified_partners is not None:
            error += float(
                np.sum(np.abs(modified_partners - original_partners) ** 2)
            )

        if catalogue.transform == "dft":
            error *= 2.0

        embedding_cost = math.log10(error / n_calibration + 1e-6)

        features.append(
            [energy, entropy, math.log1p(local_variance),
             skewness, kurtosis, embedding_cost]
        )

        if band["kind"] == "ring":
            positions.append(positional_encoding_ring(band["rho"]))
        else:
            positions.append(
                positional_encoding_subband(band["level"], band["btype"])
            )

    raw_features = np.asarray(features, dtype=np.float32)

    raw_features = (
        raw_features - raw_features.mean(axis=0, keepdims=True)
    ) / (raw_features.std(axis=0, keepdims=True) + 1e-8)

    return np.concatenate(
        [raw_features, np.asarray(positions, dtype=np.float32)], axis=1
    )


def build_band_inputs(algorithm, catalogue, luminance, calibration_bits, transform_cache):
    transform_name = catalogue.transform

    if transform_name not in transform_cache:
        transform_cache[transform_name] = transform_forward(
            transform_name, luminance
        ).reshape(-1)

    return compute_band_descriptors(
        algorithm, catalogue, transform_cache[transform_name], calibration_bits
    )


# %% SABSNet
class ResidualPerceptron(nn.Module):
    def __init__(self, d_in, h_dim):
        super().__init__()

        self.input_projection = nn.Linear(d_in, h_dim)
        self.layer1 = nn.Linear(h_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        x = self.input_projection(x)

        return x + self.layer2(F.gelu(self.layer1(x)))


class MHSA(nn.Module):
    def __init__(self, h_dim, n_heads, d_head):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        self.query = nn.Linear(h_dim, n_heads * d_head, bias=False)
        self.key = nn.Linear(h_dim, n_heads * d_head, bias=False)
        self.value = nn.Linear(h_dim, n_heads * d_head, bias=False)
        self.output = nn.Linear(n_heads * d_head, h_dim)

    def forward(self, tokens):
        batch, n_tokens, _ = tokens.shape

        def split(x):
            return x.view(batch, n_tokens, self.n_heads, self.d_head).transpose(1, 2)

        queries = split(self.query(tokens))
        keys = split(self.key(tokens))
        values = split(self.value(tokens))

        attention = torch.softmax(
            (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_head), dim=-1
        )

        output = attention @ values

        output = output.transpose(1, 2).contiguous().view(
            batch, n_tokens, self.n_heads * self.d_head
        )

        return self.output(output)


class SABSNet(nn.Module):
    def __init__(self, d_in=D_IN, h_dim=H_DIM, n_heads=N_HEADS, d_head=D_HEAD,
                 n_algorithms=len(ALGO_LIST)):
        super().__init__()

        self.encoder = ResidualPerceptron(d_in, h_dim)
        self.attention = MHSA(h_dim, n_heads, d_head)
        self.algorithm_embedding = nn.Embedding(n_algorithms, h_dim)

        self.selector = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, 1),
        )

        self.reconstruction = nn.Linear(h_dim, D_RAW)

    def encode(self, inputs):
        encoded = self.encoder(inputs)
        contextualized = self.attention(encoded) + encoded

        return encoded, contextualized

    def forward(self, inputs, algorithm_indices):
        encoded, contextualized = self.encode(inputs)

        conditioned = contextualized + self.algorithm_embedding(
            algorithm_indices
        ).unsqueeze(1)

        scores = self.selector(conditioned).squeeze(-1)

        return scores, conditioned, encoded, contextualized


def diversity_loss(tokens):
    normalized = F.normalize(tokens[0], dim=-1)

    n_bands = normalized.shape[0]

    if n_bands < 2:
        return tokens.new_zeros(())

    correlation = normalized @ normalized.T

    off_diagonal = correlation - torch.eye(n_bands, device=correlation.device)

    return (off_diagonal ** 2).sum() / (n_bands * (n_bands - 1))


# %% Band selection
def repair_subset(order, k, capacities, required_bits):
    selected = list(order[:k])
    remaining = list(order[k:])

    while capacities[selected].sum() < required_bits and remaining:
        selected.append(remaining.pop(0))

    if capacities[selected].sum() < required_bits:
        raise ValueError("Selected bands do not have enough capacity.")

    return sorted(int(index) for index in selected)


def select_bands(scores, k, capacities, required_bits):
    order = np.argsort(-np.asarray(scores), kind="stable")

    return repair_subset(order, k, capacities, required_bits)


def sample_candidate_subset(n_bands, k, capacities, required_bits, rng):
    return repair_subset(rng.permutation(n_bands), k, capacities, required_bits)


# %% SABSNet training
def make_inputs(cover_path, algorithm, catalogues, calibration_bits):
    rgb = load_cover(cover_path)

    inputs = build_band_inputs(
        algorithm,
        catalogues[algorithm],
        rgb_to_luma(rgb),
        calibration_bits,
        transform_cache={},
    )

    return rgb, inputs


def stage1_pretrain(model, cover_paths, catalogues, calibration_bits,
                    epochs=STAGE1_EPOCHS, mask_fraction=0.30,
                    learning_rate=1e-3, max_samples=STAGE1_MAX_SAMPLES):
    rng = np.random.default_rng(RANDOM_SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    paths = cover_paths[:max_samples]

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for cover_path in tqdm(paths, desc=f"Stage-I epoch {epoch + 1}/{epochs}"):
            algorithm = ALGO_LIST[int(rng.integers(len(ALGO_LIST)))]

            _, descriptors = make_inputs(
                cover_path, algorithm, catalogues, calibration_bits
            )

            inputs = torch.tensor(
                descriptors, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            n_bands = inputs.shape[1]
            n_masked = max(1, int(mask_fraction * n_bands))

            masked_indices = torch.as_tensor(
                rng.permutation(n_bands)[:n_masked], dtype=torch.long, device=DEVICE
            )

            masked_inputs = inputs.clone()
            masked_inputs[0, masked_indices, :D_RAW] = 0.0

            _, contextualized = model.encode(masked_inputs)

            reconstruction = model.reconstruction(contextualized)

            loss = LAMBDA_SSL * F.mse_loss(
                reconstruction[0, masked_indices],
                inputs[0, masked_indices, :D_RAW],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(
            f"  [Stage-I] epoch {epoch + 1}: "
            f"mean SSL loss = {total_loss / max(1, len(paths)):.6f}"
        )

    return model


def stage2_train(model, cover_paths, catalogues, payload_length, cover_directory,
                 min_psnr_gap, epochs=STAGE2_EPOCHS,
                 candidate_subsets=STAGE2_CANDIDATE_SUBSETS, k_fraction=K_FRAC,
                 margin=0.5, learning_rate=5e-4, max_samples=STAGE2_MAX_SAMPLES):
    rng = np.random.default_rng(RANDOM_SEED + 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    paths = cover_paths[:max_samples]

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_updates = 0

        for cover_path in tqdm(paths, desc=f"Stage-II epoch {epoch + 1}/{epochs}"):
            algorithm = ALGO_LIST[int(rng.integers(len(ALGO_LIST)))]

            catalogue = catalogues[algorithm]

            image_id = image_id_from_path(cover_path, cover_directory)

            payload_bits, _ = random_payload_for_instance(
                image_id, algorithm, payload_length
            )

            rgb, descriptors = make_inputs(
                cover_path, algorithm, catalogues, payload_bits
            )

            n_bands = descriptors.shape[0]

            k = max(1, int(k_fraction * n_bands))

            inputs = torch.tensor(
                descriptors, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            algorithm_index = torch.tensor(
                [ALGO_INDEX[algorithm]], dtype=torch.long, device=DEVICE
            )

            scores, conditioned, _, _ = model(inputs, algorithm_index)

            subsets = {
                tuple(
                    sample_candidate_subset(
                        n_bands, k, catalogue.capacities, len(payload_bits), rng
                    )
                )
                for _ in range(candidate_subsets)
            }

            labels = []
            subset_scores = []

            for subset in subsets:
                try:
                    result = embed_image(
                        algorithm, rgb, payload_bits, list(subset), catalogue
                    )
                except Exception:
                    continue

                psnr = compute_psnr(rgb, result["stego"])

                target = 99.0 if not np.isfinite(psnr) else psnr

                if result["ber"] > 0.0:
                    target -= 100.0

                labels.append(target)

                subset_scores.append(scores[0, list(subset)].mean())

            ranking_terms = []

            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[i] > labels[j] + min_psnr_gap:
                        ranking_terms.append(
                            torch.clamp(
                                margin - subset_scores[i] + subset_scores[j],
                                min=0.0,
                            )
                        )

            if not ranking_terms:
                continue

            ranking = torch.stack(ranking_terms).mean()

            loss = LAMBDA_RANK * ranking + LAMBDA_DIV * diversity_loss(conditioned)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_updates += 1

        if n_updates:
            print(
                f"  [Stage-II] epoch {epoch + 1}: "
                f"mean total loss = {total_loss / n_updates:.6f}"
            )
        else:
            print(f"  [Stage-II] epoch {epoch + 1}: no ranking updates.")

    return model


def get_selected_bands(model, descriptors, algorithm, catalogue, payload_length):
    model.eval()

    n_bands = descriptors.shape[0]

    k = max(1, int(K_FRAC * n_bands))

    with torch.no_grad():
        inputs = torch.tensor(
            descriptors, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        algorithm_index = torch.tensor(
            [ALGO_INDEX[algorithm]], dtype=torch.long, device=DEVICE
        )

        scores, _, _, _ = model(inputs, algorithm_index)

    score_vector = scores[0].cpu().numpy()

    selected_bands = select_bands(score_vector, k, catalogue.capacities, payload_length)

    return selected_bands, score_vector


# %% Metadata and per-image processing
def sanitize_metadata_row(row):
    allowed = (
        "image_id", "algorithm", "aux",
        "psnr", "snr", "ssim", "mae", "mse",
        "ber", "recovered_ok",
    )

    output = {key: row[key] for key in allowed if key in row}

    if "error" in row:
        output["error"] = row["error"]

    return output


def dump_metadata(metadata_by_algorithm, algorithm_dirs):
    for algorithm, rows in metadata_by_algorithm.items():
        metadata_path = algorithm_dirs[algorithm] / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)


def process_image(cover_path, image_id, payload_length, model, catalogues, algorithm_dirs):
    cover = load_cover(cover_path)
    luminance = rgb_to_luma(cover)

    transform_cache = {}
    results = {}

    for algorithm in ALGO_LIST:
        try:
            payload_bits, payload_seed = random_payload_for_instance(
                image_id, algorithm, payload_length
            )

            catalogue = catalogues[algorithm]

            descriptors = build_band_inputs(
                algorithm, catalogue, luminance, payload_bits, transform_cache
            )

            selected_bands, scores = get_selected_bands(
                model, descriptors, algorithm, catalogue, len(payload_bits)
            )

            result = embed_image(
                algorithm, cover, payload_bits, selected_bands, catalogue
            )

            output_path = algorithm_dirs[algorithm] / f"{image_id}_spatial.png"

            Image.fromarray(result["stego"]).save(output_path)

            saved_stego = np.array(
                Image.open(output_path).convert("RGB"), dtype=np.uint8
            )

            decoded_channels = decode_image(
                algorithm, saved_stego, len(payload_bits), selected_bands,
                catalogue, result["delta"], result["skip"],
            )

            ber = max(
                compute_ber(payload_bits, decoded) for decoded in decoded_channels
            )

            metrics = compute_image_metrics(cover, saved_stego)

            results[algorithm] = sanitize_metadata_row(
                {
                    "image_id": image_id,
                    "algorithm": algorithm,
                    "aux": {
                        "payload_mode": "per_image_algorithm_random",
                        "payload_length_bits": int(len(payload_bits)),
                        "payload_master_seed": int(PAYLOAD_MASTER_SEED),
                        "payload_seed": int(payload_seed),
                        "payload_sha256": payload_sha256(payload_bits),
                        "band_ids": selected_bands,
                        "band_scores": [round(float(s), 6) for s in scores],
                        "delta": result["delta"],
                        "skip": result["skip"],
                        "n_boosted_pairs": result["n_boosted"],
                    },
                    "psnr": round_or_none(metrics["psnr"], 4),
                    "snr": round_or_none(metrics["snr"], 4),
                    "ssim": round_or_none(metrics["ssim"], 6),
                    "mae": round_or_none(metrics["mae"], 6),
                    "mse": round_or_none(metrics["mse"], 6),
                    "ber": round(ber, 8),
                    "recovered_ok": bool(ber == 0.0),
                }
            )

        except Exception as exception:
            results[algorithm] = sanitize_metadata_row(
                {
                    "image_id": image_id,
                    "algorithm": algorithm,
                    "error": str(exception),
                }
            )

    return results


# %% CSV export
CSV_COLUMNS = [
    "level", "target_bpp", "image_id", "algorithm",
    "psnr", "snr", "ssim", "mae", "mse",
    "ber", "recovered_ok", "delta", "n_boosted_pairs", "error",
]


def flatten_rows(level, target_bpp, metadata_by_algorithm):
    records = []

    for algorithm, rows in metadata_by_algorithm.items():
        for row in rows:
            aux = row.get("aux", {})

            records.append(
                {
                    "level": level,
                    "target_bpp": target_bpp,
                    "image_id": row.get("image_id"),
                    "algorithm": algorithm,
                    "psnr": row.get("psnr"),
                    "snr": row.get("snr"),
                    "ssim": row.get("ssim"),
                    "mae": row.get("mae"),
                    "mse": row.get("mse"),
                    "ber": row.get("ber"),
                    "recovered_ok": row.get("recovered_ok"),
                    "delta": aux.get("delta"),
                    "n_boosted_pairs": aux.get("n_boosted_pairs"),
                    "error": row.get("error"),
                }
            )

    return records


def write_metrics_csv(records, csv_path=METRICS_CSV):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(records, columns=CSV_COLUMNS)

    frame.to_csv(csv_path, index=False, encoding="utf-8")

    return frame


# %% Main pipeline
def main(
    level,
    cover_directory=COVER_DIR,
    output_base=OUTPUT_BASE,
    max_images=MAX_IMAGES,
    stage1_samples=STAGE1_MAX_SAMPLES,
    stage2_samples=STAGE2_MAX_SAMPLES,
    stage1_epochs=STAGE1_EPOCHS,
    stage2_epochs=STAGE2_EPOCHS,
):
    target_bpp = LEVELS[level]["target_bpp"]
    min_psnr_gap = LEVELS[level]["min_psnr_gap"]
    output_root = Path(output_base) / level

    set_seeds()

    algorithm_dirs = make_algorithm_dirs(output_root)

    cover_paths = collect_cover_paths(cover_directory, max_images)

    if not cover_paths:
        raise FileNotFoundError(f"No images found in {cover_directory}")

    image_width, image_height = IMAGE_SIZE

    payload_length = int(target_bpp * image_height * image_width)

    calibration_rng = np.random.default_rng(RANDOM_SEED)

    calibration_bits = calibration_rng.integers(
        low=0, high=2, size=CAL_BITS, dtype=np.uint8
    )

    catalogues = build_catalogues(image_height, image_width)

    print("=" * 80)
    print(f"Level           : {level}")
    print(f"Cover directory : {cover_directory}")
    print(f"Output root     : {output_root}")
    print(f"Cover images    : {len(cover_paths)}")
    print(f"Resolution      : {image_width} x {image_height}")
    print(f"Payload         : {payload_length} bits/channel")
    print(f"Target rate     : {target_bpp} bpp/channel")
    print(f"Min PSNR gap    : {min_psnr_gap}")
    print("Payload mode    : independent Bernoulli(0.5) payload per image/algorithm")
    print(f"Payload seed    : {PAYLOAD_MASTER_SEED}")
    print(f"Device          : {DEVICE}")
    print("=" * 80)

    print("\nBand capacities:")

    for algorithm in ALGO_LIST:
        capacities = catalogues[algorithm].capacities

        print(
            f"  {algorithm:<12} "
            f"bands={len(capacities):2d}, "
            f"minimum={capacities.min():6d}, "
            f"maximum={capacities.max():6d}"
        )

    run_config = {
        "level": level,
        "cover_directory": str(cover_directory),
        "output_root": str(output_root),
        "image_size": IMAGE_SIZE,
        "target_bpp_per_channel": target_bpp,
        "min_psnr_gap": min_psnr_gap,
        "payload_mode": "per_image_algorithm_random",
        "payload_length_bits_per_channel": payload_length,
        "payload_master_seed": PAYLOAD_MASTER_SEED,
        "payload_generation": (
            "iid Bernoulli(0.5), deterministic SHA-256 seed "
            "from master_seed|image_id|algorithm"
        ),
        "n_rings": N_RINGS,
        "k_fraction": K_FRAC,
        "base_delta": BASE_DELTA,
        "delta_scales": DELTA_SCALES,
        "phase_step": PHASE_STEP,
        "cross_margin_multiplier": CROSS_MARGIN,
        "max_closed_loop_rounds": MAX_ROUNDS,
        "calibration_bits": CAL_BITS,
        "random_seed": RANDOM_SEED,
        "metrics": list(METRIC_KEYS) + ["ber"],
        "algorithms": ALGO_SPEC,
    }

    with open(output_root / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    model = SABSNet().to(DEVICE)

    print("\n=== Stage I: self-supervised spectral representation learning ===")

    model = stage1_pretrain(
        model=model,
        cover_paths=cover_paths,
        catalogues=catalogues,
        calibration_bits=calibration_bits,
        epochs=stage1_epochs,
        max_samples=min(stage1_samples, len(cover_paths)),
    )

    print("\n=== Stage II: ranking-based semi-supervised band selection ===")

    model = stage2_train(
        model=model,
        cover_paths=cover_paths,
        catalogues=catalogues,
        payload_length=payload_length,
        cover_directory=cover_directory,
        min_psnr_gap=min_psnr_gap,
        epochs=stage2_epochs,
        max_samples=min(stage2_samples, len(cover_paths)),
    )

    weights_path = output_root / "sabsnet_weights.pt"

    torch.save(model.state_dict(), weights_path)

    print(f"\nSABSNet weights saved to: {weights_path}")

    metadata_by_algorithm = {algorithm: [] for algorithm in ALGO_LIST}

    for index, cover_path in enumerate(
        tqdm(cover_paths, desc=f"Generating stego images [{level}]"), start=1
    ):
        image_id = image_id_from_path(cover_path, cover_directory)

        try:
            results = process_image(
                cover_path=cover_path,
                image_id=image_id,
                payload_length=payload_length,
                model=model,
                catalogues=catalogues,
                algorithm_dirs=algorithm_dirs,
            )

            for algorithm, row in results.items():
                metadata_by_algorithm[algorithm].append(row)

        except Exception as exception:
            for algorithm in ALGO_LIST:
                metadata_by_algorithm[algorithm].append(
                    {
                        "image_id": image_id,
                        "algorithm": algorithm,
                        "error": f"Image processing failed: {exception}",
                    }
                )

        if index % 500 == 0:
            dump_metadata(metadata_by_algorithm, algorithm_dirs)

    dump_metadata(metadata_by_algorithm, algorithm_dirs)

    print("\n" + "=" * 80)
    print(f"SUMMARY [{level}]")
    print("=" * 80)

    def finite_mean(rows, key):
        values = [
            row[key] for row in rows
            if row.get(key) is not None and np.isfinite(row[key])
        ]

        return float(np.mean(values)) if values else float("nan")

    for algorithm in ALGO_LIST:
        rows = metadata_by_algorithm[algorithm]

        successful = [row for row in rows if "ber" in row]
        errors = [row for row in rows if "error" in row]
        exact = [row for row in successful if row.get("recovered_ok", False)]

        print(f"\n[{algorithm}]")
        print(f"  Processed successfully : {len(successful)}/{len(rows)}")
        print(f"  Errors                 : {len(errors)}")
        print(f"  Exact recovery         : {len(exact)}/{len(successful)}")
        print(f"  Mean PSNR              : {finite_mean(successful, 'psnr'):.4f} dB")
        print(f"  Mean SNR               : {finite_mean(successful, 'snr'):.4f} dB")
        print(f"  Mean SSIM              : {finite_mean(successful, 'ssim'):.6f}")
        print(f"  Mean MAE               : {finite_mean(successful, 'mae'):.6f}")
        print(f"  Mean MSE               : {finite_mean(successful, 'mse'):.6f}")
        print(f"  Mean BER               : {finite_mean(successful, 'ber'):.8f}")
        print(f"  Output directory       : {algorithm_dirs[algorithm]}")

    return metadata_by_algorithm


# %% Runs all payload levels and export one combined CSV
if __name__ == "__main__":
    all_metadata = {}
    all_records = []

    for level in LEVELS:  # "low", "modest", "high"
        all_metadata[level] = main(
            level=level,
                max_images=1000,
                stage1_samples=700,
                stage2_samples=400,
                stage1_epochs=5,
                stage2_epochs=4,
        )

        all_records.extend(
            flatten_rows(
                level, LEVELS[level]["target_bpp"], all_metadata[level]
            )
        )

        metrics_frame = write_metrics_csv(all_records)

    print(f"\nCombined metrics CSV saved to: {METRICS_CSV}")
    print(f"Rows: {len(metrics_frame)}")