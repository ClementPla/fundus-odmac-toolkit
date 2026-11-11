"""Upload the best checkpoint of each local training run to a branch of its Hugging Face repo.

Uses the credentials of `huggingface-cli login`. The branch is created from `main` if needed,
so `main` (the previous weights) is left untouched.

Usage:
    python scripts/upload_to_hf_hub.py --checkpoints checkpoints/OpticDisk-Macula-Segmentation-v2 --branch v2
"""

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
from fundus_odmac_toolkit.models.hf_hub import ROOT_HF, HuggingFaceModel  # noqa: E402
from run_benchmarks import find_local_checkpoints  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--branch", default="v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    for (arch, encoder), ckpt in find_local_checkpoints(args.checkpoints).items():
        repo_id = f"{ROOT_HF}{arch}-{encoder}"
        hf_model = HuggingFaceModel(arch=arch, encoder=encoder, classes=3)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        hf_model.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
        print(f"{repo_id}@{args.branch} <- {ckpt}")
        if args.dry_run:
            continue
        api.create_branch(repo_id, branch=args.branch, exist_ok=True)
        hf_model.push_to_hub(
            repo_id,
            branch=args.branch,
            commit_message=f"Retrained weights ({ckpt.parent.name}/{ckpt.name}): Dice + CE loss, AdamW with "
            "per-part learning rates, warm-up + cosine schedule",
        )


if __name__ == "__main__":
    main()
