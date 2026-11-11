"""Evaluate every published OD/macula model on all benchmarks (no test-time augmentation).

Benchmarks:
    - IDRiD segmentation challenge, optic disc: IoU (%) on the 27 test images
    - RIGA Magrabia / MESSIDOR, optic disc: IoU (%) against the majority vote of the 6 annotators
    - IDRiD localization challenge, optic disc and fovea: mean Euclidean distance (px) between the
      centroid of the predicted region and the ground-truth center, on the 103 test images.
      When no pixel is assigned to the structure, the pixel with the highest probability for it is
      used instead, so every image is scored; `missed` counts these images and `value_detected`
      is the mean over the other ones only.
      Also reported after `postprocess_od_macula` ("*_distance_pp"), which keeps only the most
      plausible OD / macula region (images where it finds none are excluded and counted in `missed`).

IoU is accumulated over all pixels of the test set (torchmetrics binary JaccardIndex).

Usage:
    python benchmarks/run_benchmarks.py [--data /path/to/data] [--checkpoints DIR] [--out results.jsonl]

By default the published Hugging Face weights are evaluated. With --checkpoints, the best checkpoint
of each local training run in DIR (see scripts/train_queue.sh) is evaluated instead.
Results are written to benchmarks/<out> (one line per model and benchmark).
"""

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchmetrics
from fundus_data_toolkit.functional import open_image
import fundus_odmac_toolkit.models.segmentation as seg
import torchseg
from fundus_odmac_toolkit.models.hf_hub import list_models
from fundus_odmac_toolkit.models.segmentation import postprocess_od_macula, segment
from tqdm import tqdm

OD, MACULA = 1, 2
HERE = Path(__file__).parent


def predict_probs(image, arch, encoder):
    return segment(image, use_tta=False, autofit_resolution=True, reverse_autofit=True, arch=arch, encoder=encoder)


def predict(image, arch, encoder):
    return predict_probs(image, arch, encoder).argmax(0)


def find_local_checkpoints(ckpt_dir: Path) -> dict:
    """Map (arch, encoder) -> best checkpoint, read from the training logs of scripts/train_queue.sh."""
    found = {}
    for log in sorted((ckpt_dir / "logs").glob("*-*.log")):
        if log.stem.startswith("queue"):
            continue
        arch, encoder = log.stem.split("-", 1)
        best = re.findall(r"Restoring states from the checkpoint path at (\S+\.ckpt)", log.read_text(errors="ignore"))
        if best:
            found[(arch, encoder)] = Path(best[-1])
    return found


def use_local_checkpoints(checkpoints: dict):
    """Make segment() load local checkpoints instead of downloading from Hugging Face."""

    def load(arch, encoder):
        model = torchseg.create_model(arch, encoder, encoder_weights=None, classes=3)
        state = torch.load(checkpoints[(arch, encoder)], map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
        return model

    seg.download_model = load
    seg.get_model.cache_clear()


# ---------------------------------------------------------------- RIGA
def extract_mask(img_gt, ref_img):
    """The RIGA annotations are drawn on top of the image: recover the filled contour."""
    mask_border = np.abs(img_gt - ref_img).max(-1) > 25
    return cv2.floodFill(mask_border.astype(np.uint8), None, (0, 0), 255)[1]


def load_riga(root: Path):
    images, masks = {}, {}
    for img_gt in tqdm(sorted(root.glob("**/*-*.tif")), desc=f"Loading {root.name}"):
        img_name = img_gt.stem.split("-")[0]
        parent = img_gt.parent
        img_id = f"{parent.stem}/{img_name.lower()}"
        if img_id in images:
            continue
        # File names are not standardized in RIGA
        img_path = (
            list(parent.glob(f"{img_name}prime.tif"))
            + list(parent.glob(f"{img_name.capitalize()}prime.tif"))
            + list(parent.glob(f"{img_name.capitalize()}.tif"))
            + list(parent.glob(f"{img_name}.tif"))
        )
        if not img_path:
            print(f"No image found for {img_id}")
            continue
        annotations = list(parent.glob(f"{img_name}-*.tif")) + list(parent.glob(f"{img_name.capitalize()}-*.tif"))
        try:
            img = open_image(img_path[0])
        except Exception:
            print(f"Failed to open {img_path[0]}")
            continue
        gts = np.stack([extract_mask(img, open_image(f)) for f in annotations])
        images[img_id] = img
        masks[img_id] = 1 - (np.mean(gts, axis=0) > gts.shape[0] / 2)  # majority vote
    return images, masks


def riga_iou(images, masks, arch, encoder):
    jaccard = torchmetrics.JaccardIndex(task="binary").cuda()
    for img_id, img in images.items():
        gt = torch.from_numpy(masks[img_id]).cuda() > 0
        jaccard.update((predict(img, arch, encoder) == OD).long(), gt)
    return jaccard.compute().item() * 100


# ---------------------------------------------------------------- IDRiD
def idrid_iou(root: Path, arch, encoder):
    img_dir = root / "A. Segmentation/1. Original Images/b. Testing Set"
    mask_dir = root / "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc"
    jaccard = torchmetrics.JaccardIndex(task="binary").cuda()
    for f in sorted(img_dir.glob("*.jpg")):
        mask = cv2.imread(str(mask_dir / f"{f.stem}_OD.tif"), cv2.IMREAD_GRAYSCALE) > 0
        jaccard.update((predict(open_image(f), arch, encoder) == OD).long(), torch.from_numpy(mask).cuda())
    return jaccard.compute().item() * 100


def idrid_localization(root: Path, arch, encoder):
    img_dir = root / "C. Localization/1. Original Images/b. Testing Set"
    gt_dir = root / "C. Localization/2. Groundtruths"
    gts = {
        OD: pd.read_csv(gt_dir / "1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv"),
        MACULA: pd.read_csv(gt_dir / "2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv"),
    }
    names = {OD: "idrid_od_distance", MACULA: "idrid_fovea_distance"}
    keys = [f"{n}{s}" for n in names.values() for s in ("", "_pp")]
    distances = {k: [] for k in keys}  # (distance, detected)
    files = sorted(img_dir.glob("*.jpg"))
    for f in files:
        probs = predict_probs(open_image(f), arch, encoder)
        pred = probs.argmax(0)
        post = postprocess_od_macula(probs.cpu().numpy())
        for label, gt in gts.items():
            gx, gy = gt[gt["Image No"] == f.stem][["X- Coordinate", "Y - Coordinate"]].values[0]
            y, x = (pred == label).nonzero().float().mean(dim=0).tolist()
            detected = not math.isnan(x)
            if not detected:  # fall back on the most probable pixel
                y, x = np.unravel_index(int(probs[label].argmax()), probs.shape[1:])
            distances[names[label]].append((math.hypot(x - gx, y - gy), detected))
            center = post["od_center" if label == OD else "macula_center"]
            if center is None:
                distances[names[label] + "_pp"].append((float("nan"), False))
            else:
                distances[names[label] + "_pp"].append((math.hypot(center[0] - gx, center[1] - gy), True))
    out = {}
    for key, d in distances.items():
        all_d = np.array([v for v, _ in d])
        det = np.array([v for v, ok in d if ok])
        missed = len(d) - len(det)
        value = float(np.mean(det)) if key.endswith("_pp") else float(np.mean(all_d))
        out[key] = {"value": value, "missed": missed, "n": len(files)}
        if not key.endswith("_pp"):
            out[key]["value_detected"] = float(np.mean(det)) if len(det) else float("nan")
    return out


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/clement/Documents/data"))
    parser.add_argument("--checkpoints", type=Path, help="Evaluate local training runs instead of HF weights")
    parser.add_argument("--out", default="results.jsonl", help="Output file name, inside benchmarks/")
    parser.add_argument("--revision", help="Hugging Face branch to evaluate (default: the toolkit default)")
    parser.add_argument(
        "--only",
        choices=["idrid", "riga"],
        help="Run only these benchmarks and replace their lines in --out, keeping the others",
    )
    args = parser.parse_args()
    if args.revision:
        seg.DEFAULT_REVISION = args.revision

    if args.checkpoints:
        checkpoints = find_local_checkpoints(args.checkpoints)
        use_local_checkpoints(checkpoints)
        models = list(checkpoints)
        for (arch, encoder), path in checkpoints.items():
            print(f"{arch}-{encoder}: {path}")
    else:
        models = [tuple(m) for m in list_models()]
    OUT = HERE / args.out
    keep = []
    if args.only and OUT.exists():
        prefix = "idrid_" if args.only == "idrid" else "riga_"
        keep = [line for line in OUT.read_text().splitlines() if not json.loads(line)["benchmark"].startswith(prefix)]
    OUT.write_text("".join(line + "\n" for line in keep))

    def log(benchmark, arch, encoder, value, **extra):
        row = {"benchmark": benchmark, "arch": arch, "encoder": encoder, "value": value, **extra}
        print(row, flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")

    idrid = args.data / "IDRID"
    for arch, encoder in tqdm(models if args.only != "riga" else [], desc="IDRiD"):
        log("idrid_od_iou", arch, encoder, idrid_iou(idrid, arch, encoder), n=27)
        for name, res in idrid_localization(idrid, arch, encoder).items():
            log(name, arch, encoder, **res)

    for subset in ["Magrabia", "MESSIDOR"] if args.only != "idrid" else []:
        images, masks = load_riga(args.data / "RIGA-dataset" / subset)
        for arch, encoder in tqdm(models, desc=subset):
            log(f"riga_{subset.lower()}_od_iou", arch, encoder, riga_iou(images, masks, arch, encoder), n=len(images))
        del images, masks


if __name__ == "__main__":
    main()
