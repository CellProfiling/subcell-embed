"""Write a small synthetic dataset in the MDS format expected by data/dataset.py.

Each MDS sample corresponds to one antibody and holds a stack of single-cell
crops with five channels (red: microtubules, yellow: ER, blue: nucleus,
green: protein of interest, and a binary cell mask). The protein channel is
either nuclear ("Nucleoplasm") or cytoplasmic ("Cytosol"), so the online
localization classifier has a learnable signal.

Usage (from the repository root):
    python scripts/make_demo_data.py --out demo_data/hpa_mds_448
"""

import argparse
import os

import numpy as np
import pandas as pd
from streaming import MDSWriter

COLUMNS = {
    "antibody": "str",
    "cell_line": "str",
    "plate_position": "str",
    "ensembl_ids": "str",
    "locations": "str",
    "img": "ndarray:uint16",
    "targets": "ndarray:uint8",
    "bboxes": "ndarray:uint16",
}
CELL_LINES = ["U2OS", "A-431", "HeLa", "MCF7"]
LOCATIONS = ["Nucleoplasm", "Cytosol"]


def get_categories(annotation_path):
    location_map = pd.read_csv(annotation_path, sep="\t")
    categories = location_map["Original annotation"].unique().tolist()
    categories.append("Negative")
    return categories


def ellipse(yy, xx, cy, cx, ry, rx, angle):
    cos, sin = np.cos(angle), np.sin(angle)
    y, x = yy - cy, xx - cx
    return ((x * cos + y * sin) / rx) ** 2 + ((-x * sin + y * cos) / ry) ** 2 <= 1.0


def make_cell(rng, size, location):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = size / 2 + rng.uniform(-size / 16, size / 16, 2)
    angle = rng.uniform(0, np.pi)
    r_cell = rng.uniform(0.25, 0.4, 2) * size
    r_nuc = r_cell * rng.uniform(0.35, 0.5)

    cell = ellipse(yy, xx, c[0], c[1], r_cell[0], r_cell[1], angle)
    nucleus = ellipse(yy, xx, c[0], c[1], r_nuc[0], r_nuc[1], angle)
    cytoplasm = cell & ~nucleus

    def noisy(region, level):
        signal = region * level * rng.uniform(0.6, 1.0)
        signal = signal + rng.normal(0, 0.05, region.shape) * region
        return np.clip(signal, 0, 1)

    red = noisy(cytoplasm, 0.6)
    yellow = noisy(cytoplasm, 0.8)
    blue = noisy(nucleus, 1.0)
    green = noisy(nucleus if location == "Nucleoplasm" else cytoplasm, 1.0)

    img = np.stack([red, yellow, blue, green], axis=-1) * 65535
    img = np.concatenate([img, cell[..., None].astype(np.float32)], axis=-1)
    return img.astype(np.uint16)


def make_sample(rng, idx, n_cells, size, categories):
    location = LOCATIONS[idx % len(LOCATIONS)]
    cell_lines = rng.choice(CELL_LINES, n_cells)
    plate_positions = [f"{1000 + idx}_A{(i % 4) + 1}_1" for i in range(n_cells)]

    targets = np.zeros((n_cells, len(categories)), dtype=np.uint8)
    targets[:, categories.index(location)] = 1

    return {
        "antibody": f"HPA{idx:06d}",
        "cell_line": ";".join(cell_lines),
        "plate_position": ";".join(plate_positions),
        "ensembl_ids": ";".join([f"ENSG{idx:011d}"] * n_cells),
        "locations": ";".join([location] * n_cells),
        "img": np.stack([make_cell(rng, size, location) for _ in range(n_cells)]),
        "targets": targets,
        "bboxes": np.tile(np.array([0, 0, size, size], dtype=np.uint16), (n_cells, 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo_data/hpa_mds_448")
    parser.add_argument("--n_train", type=int, default=32)
    parser.add_argument("--n_valid", type=int, default=8)
    parser.add_argument("--cells_per_sample", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--annotations", default="annotations/location_group_mapping.tsv"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    categories = get_categories(args.annotations)

    offset = 0
    for split, n_samples in [("train", args.n_train), ("valid", args.n_valid)]:
        out = os.path.join(args.out, split)
        with MDSWriter(out=out, columns=COLUMNS, compression="zstd") as writer:
            for idx in range(offset, offset + n_samples):
                writer.write(
                    make_sample(
                        rng, idx, args.cells_per_sample, args.image_size, categories
                    )
                )
        offset += n_samples
        print(f"Wrote {n_samples} samples to {out}")


if __name__ == "__main__":
    main()
